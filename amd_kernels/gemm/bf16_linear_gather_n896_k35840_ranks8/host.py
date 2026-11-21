# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the sharded drafter context projection: tier choice, launches, warm-up."""
import torch
from amd_kernels import warm_up

from . import kernel as K1
from .comm import MAX_ROWS, SLAB_SPLIT, ProjComm
from .ipc import f32_bits, run_compiled
from .pack import pack_rows, pack_weight
from .pack_rows import compile_pack_rows
from .reduce_gather import compile_reduce_gather
from .reference import K, N_FULL, N_SHARD

NAME = "gemm/bf16_linear_gather_n896_k35840_ranks8"
KSTEPS = K // 32

# (rows up to, (mt, g, split, depth, wr, wk)): activation row tiles per workgroup, weight groups, K split, loads in
# flight, wave grid (rows x K)
TIERS = (
    (16, (1, 4, 16, 4, 1, 2)),
    (32, (2, 4, 16, 4, 1, 2)),
    (48, (3, 4, 16, 4, 1, 2)),
    (80, (5, 4, 16, 4, 1, 2)),
    (128, (8, 4, 16, 4, 1, 2)),
    (192, (12, 7, 32, 4, 2, 1)),
)

assert max(cfg[2] for _, cfg in TIERS) <= SLAB_SPLIT


def config_for(rows: int) -> tuple:
    for limit, cfg in TIERS:
        if rows <= limit:
            return cfg
    raise ValueError(f"at most {TIERS[-1][0]} rows")


def packed_tiles(rows: int) -> int:
    """16-row tiles of the packed activations the kernel reads for ``rows`` rows."""
    mt = config_for(rows)[0]
    return -(-rows // (16 * mt)) * mt


def packed_rows_buffer(rows: int, device) -> torch.Tensor:
    return torch.empty(packed_tiles(rows), KSTEPS, 64, 8, dtype=torch.bfloat16, device=device)


def _pack_mode(aux, out, rows):
    return ()


def _pack_inputs(mode, n, aux, out, rows):
    return ([torch.zeros(n, K // 5, dtype=torch.bfloat16, device=out.device) for _ in range(5)],
            packed_rows_buffer(n, out.device), n), {}


@warm_up.entry(NAME + "/pack_aux", mode=_pack_mode, sizes=lambda mode: [16], inputs=_pack_inputs)
def pack_aux(aux: list[torch.Tensor], out: torch.Tensor, rows: int) -> torch.Tensor:
    """The five aux hidden states [>= rows, 7168] -> the packed tiles of ``rows`` rows in ``out``."""
    assert len(aux) == 5 and out.numel() == packed_tiles(rows) * KSTEPS * 512
    assert all(h.dtype == torch.bfloat16 and h.stride(1) == 1 and h.stride(0) == aux[0].stride(0) for h in aux)
    run_compiled(compile_pack_rows(), out.data_ptr(), *[h.data_ptr() for h in aux], rows, aux[0].stride(0),
                 packed_tiles(rows) * KSTEPS, torch.cuda.current_stream())
    return out


def linear_shard(x_packed: torch.Tensor, w_packed: torch.Tensor, slab: torch.Tensor, rows: int) -> torch.Tensor:
    """fp32 slabs [split, rows, 896] of this rank's 896 output columns (summed in slab order by the gather)."""
    cfg = config_for(rows)
    assert slab.dtype == torch.float32 and slab.numel() >= cfg[2] * rows * N_SHARD
    run_compiled(K1.compile_linear(*cfg), x_packed.data_ptr(), w_packed.data_ptr(), slab.data_ptr(), rows,
                 torch.cuda.current_stream())
    return slab


def _mode(comm, x_packed, w_packed, norm_weight, rows, eps=1e-5, out=None):
    return ()


def _sizes(mode):
    return [limit for limit, _ in TIERS]


def _inputs(mode, n, comm, x_packed, w_packed, norm_weight, rows, eps=1e-5, out=None):
    xp = packed_rows_buffer(n, w_packed.device)
    return (comm, xp, w_packed, norm_weight, n, eps), {}


@warm_up.entry(NAME, mode=_mode, sizes=_sizes, inputs=_inputs)
def context_proj_gather_norm(comm: ProjComm, x_packed: torch.Tensor, w_packed: torch.Tensor,
                             norm_weight: torch.Tensor, rows: int, eps: float = 1e-5,
                             out: torch.Tensor | None = None) -> torch.Tensor:
    """``RMSNorm(x @ W^T)`` of the full [rows, 7168] projection on every rank (see reference.py).

    ``x_packed``: the activation tiles from ``pack_aux`` (identical on all ranks); ``w_packed``: this rank's
    ``pack_weight`` shard (rows [896 r, 896 (r + 1)) of the checkpoint weight); every rank calls with the same
    ``rows``, in the same order, on the current stream.
    """
    if not comm.bufs:
        raise RuntimeError("communicator is closed")
    assert 1 <= rows <= MAX_ROWS and norm_weight.numel() == N_FULL and norm_weight.dtype == torch.bfloat16
    cfg = config_for(rows)
    if out is None:
        out = torch.empty(rows, N_FULL, dtype=torch.bfloat16, device=w_packed.device)
    assert out.shape == (rows, N_FULL) and out.is_contiguous()
    linear_shard(x_packed, w_packed, comm.slab, rows)
    run_compiled(compile_reduce_gather(cfg[2]), rows, comm.rank, comm.slab.data_ptr(), comm.buf_table.data_ptr(),
                 comm.counts.data_ptr(), norm_weight.data_ptr(), out.data_ptr(), f32_bits(eps),
                 torch.cuda.current_stream())
    return out


__all__ = ["ProjComm", "pack_aux", "pack_rows", "pack_weight", "context_proj_gather_norm", "linear_shard"]
