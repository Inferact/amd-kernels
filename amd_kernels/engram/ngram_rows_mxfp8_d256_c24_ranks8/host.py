# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Static buffers, peer table pointers and the launch of the n-gram row pull."""
import ctypes
import functools

import torch
import torch.distributed as dist
from amd_kernels.common.hip_runtime import load_hip_runtime

from .ipc import _alloc_handle, _open_handle, _run_compiled, exchange
from .kernel import COLS, DEPTH, DIM, KS, LAYERS, RANKS, ROWS_MAX, SCALES, compile_ngram_rows
from .publish import GROUPS, compile_table_publish
from .reference import K, pad32

MAX_TILE = 192  # the largest row tile of a decode GEMM row


@functools.cache
def _kernel():
    return compile_ngram_rows()


def available() -> bool:
    """gfx950 with FlyDSL importable."""
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


class NgramRows:
    """Hash the batch's n-grams and pull the rows of both Engram layers' tables into MXFP8 GEMM inputs.

    Rank q of 8 holds rows [q * peer_rows, (q + 1) * peer_rows) of each layer's table: codes uint8 or E4M3
    [rows, 256] and E8M0 scales uint8 [rows, 8]. ``weight_ptrs[l][q]`` / ``scale_ptrs[l][q]`` are the addresses of
    rank q's shard as this process can read them; ``map_tables`` builds them over HIP IPC. The tables stay alive and
    unchanged while a graph that calls this object may replay. ``token_map`` int32 [vocab], ``multipliers``,
    ``primes``, ``offsets`` int64 [2, 4] / [2, 24] / [2, 24] on the device, ``pad_id`` the compressed pad id,
    ``image_id`` the dead token id (None: no token is dead).
    """

    def __init__(self, token_map, multipliers, primes, offsets, pad_id, peer_rows, weight_ptrs, scale_ptrs,
                 max_rows=768, image_id=None, keepalive=()):
        device = token_map.device
        assert multipliers.shape == (LAYERS, 4) and primes.shape == offsets.shape == (LAYERS, COLS)
        assert all(t.dtype == torch.int64 and t.is_contiguous() and t.device == device
                   for t in (multipliers, primes, offsets))
        assert token_map.dtype == torch.int32 and token_map.is_contiguous()
        assert 0 < max_rows <= 4096 and len(weight_ptrs) == len(scale_ptrs) == LAYERS
        self.device = device
        self.token_map = token_map
        # 1-D views: the kernel indexes these tensors linearly
        self.multipliers, self.primes, self.offsets = multipliers.view(-1), primes.view(-1), offsets.view(-1)
        self.pad_id = int(pad_id)
        self.image_id = -2 if image_id is None else int(image_id)
        self.peer_rows = int(peer_rows)
        self.max_rows = max_rows
        self._keepalive = keepalive
        flat = []
        for layer in range(LAYERS):
            assert len(weight_ptrs[layer]) == len(scale_ptrs[layer]) == RANKS
            flat += list(weight_ptrs[layer]) + list(scale_ptrs[layer])
        self.peers = torch.tensor(flat, dtype=torch.int64, device=device)
        self.code_rows = max_rows + MAX_TILE
        self.codes = torch.empty((LAYERS, self.code_rows, K), dtype=torch.uint8, device=device)
        self.scale_layer = pad32(self.code_rows) * KS
        self.scales = torch.empty((LAYERS, self.scale_layer), dtype=torch.uint8, device=device)

    @classmethod
    def map_tables(cls, weights, scales, rank, group, token_map, multipliers, primes, offsets, pad_id, peer_rows,
                   **kwargs):
        """Collective, on each of the 8 ranks after its shards reach their final allocation: publish the GPU-written
        shards, exchange HIP IPC handles and build the object. ``weights[l]`` [part_rows, 256], ``scales[l]``
        [part_rows, 8] are this rank's shards (contiguous)."""
        assert dist.get_world_size(group) == RANKS and dist.get_rank(group) == rank
        device = weights[0].device
        if torch.cuda.current_device() != device.index:
            raise ValueError("select this rank's device before mapping the tables")
        done = torch.empty(GROUPS, dtype=torch.int32, device=device)
        _run_compiled(compile_table_publish(), done, torch.cuda.current_stream())
        torch.cuda.synchronize(device)
        mine = []
        for t in (*weights, *scales):
            assert t.is_contiguous() and t.shape[1] in (DIM, SCALES)
            base, handle = _alloc_handle(t.data_ptr())
            mine.append((handle, t.data_ptr() - base))
        every = exchange(mine, rank, group)
        opened = {}
        ptrs = [[[0] * RANKS for _ in range(LAYERS)] for _ in range(2)]
        for q, entries in enumerate(every):
            for i, (raw, offset) in enumerate(entries):
                kind, layer = divmod(i, LAYERS)
                if q == rank:
                    ptrs[kind][layer][q] = (weights, scales)[kind][layer].data_ptr()
                else:
                    if raw not in opened:
                        opened[raw] = _open_handle(raw)
                    ptrs[kind][layer][q] = opened[raw] + offset
        self = cls(token_map, multipliers, primes, offsets, pad_id, peer_rows, ptrs[0], ptrs[1],
                   keepalive=(weights, scales), **kwargs)
        self.opened = opened
        self.group = group
        dist.barrier(group=group)
        return self

    def __call__(self, input_ids, positions, query_start_loc, lookback_ids, tile_rows=0):
        """Write both layers' ``wkv`` inputs of the batch's tokens; read them with ``activation``.

        ``input_ids`` int32 / int64 [M], ``positions`` int32 / int64 [M], ``query_start_loc`` int32 [R + 1] (R <= 1024
        request slots, idle ones empty), ``lookback_ids`` int32 [R, 3] (-1: unknown), all contiguous on the device.
        ``tile_rows`` 0: row-major codes; else the row tile (a multiple of 16, <= 192) of the GEMM's decode row, codes in
        the weight's (16, 16) layout, rows padded to it.
        """
        m = input_ids.numel()
        assert 0 < m <= self.max_rows and input_ids.ndim == positions.ndim == 1 and positions.numel() == m
        assert input_ids.dtype in (torch.int32, torch.int64) and positions.dtype in (torch.int32, torch.int64)
        slots = query_start_loc.numel() - 1
        assert 0 < slots <= ROWS_MAX and query_start_loc.dtype == torch.int32
        assert 0 <= tile_rows <= MAX_TILE and tile_rows % 16 == 0
        rows = -(-m // tile_rows) * tile_rows if tile_rows else m
        assert lookback_ids.dtype == torch.int32 and lookback_ids.shape == (slots, DEPTH)
        for t in (input_ids, positions, query_start_loc, lookback_ids):
            assert t.is_contiguous() and t.device == self.device
        _run_compiled(
            _kernel(), input_ids.data_ptr(), positions.data_ptr(), query_start_loc.data_ptr(), lookback_ids.data_ptr(),
            self.token_map.data_ptr(), self.multipliers, self.primes, self.offsets, self.peers, self.codes.data_ptr(),
            self.scales.data_ptr(), m, slots, input_ids.element_size() // 4, positions.element_size() // 4, DEPTH,
            self.pad_id, self.image_id, self.peer_rows, self.code_rows * K, self.scale_layer, tile_rows, rows, pad32(rows),
            torch.cuda.current_stream())

    def activation(self, layer, m, tile_rows=0):
        """(codes float8_e4m3fn [rows, 6144], scales uint8 [pad32(rows), 192]) of ``layer`` as written by the call with
        this ``tile_rows`` (rows = m, or m padded to the tile): the GEMM's activation and its preshuffled scales, views
        of the static buffers."""
        rows = -(-m // tile_rows) * tile_rows if tile_rows else m
        codes = self.codes[layer, :rows].view(torch.float8_e4m3fn)
        scales = self.scales[layer, : pad32(rows) * KS].view(pad32(rows), KS)
        return codes, scales

    def warm_up(self):
        """Compile the kernel (one variant for every dtype and row count) with an all-pad batch."""
        z = torch.zeros(1, dtype=torch.int32, device=self.device)
        self(z, z.to(torch.int64), torch.tensor([0, 1], dtype=torch.int32, device=self.device),
             torch.full((1, DEPTH), -1, dtype=torch.int32, device=self.device))
        torch.cuda.synchronize(self.device)

    def close(self):
        """Collective after every graph that reads the tables is retired (a no-op without ``map_tables``)."""
        opened = getattr(self, "opened", None)
        if opened is None:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        hip = load_hip_runtime()
        hip.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        for p in opened.values():
            rc = hip.hipIpcCloseMemHandle(ctypes.c_void_p(p))
            assert rc == 0, rc
        opened.clear()
