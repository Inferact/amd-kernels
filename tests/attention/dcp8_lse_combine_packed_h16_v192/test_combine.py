# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Contract against reference.py and bit for bit against vLLM's unpack-combine on 8 GPUs: row counts across the tiles,
eager and graph replay with changed inputs and row counts, empty shards, NaN and +inf LSEs, two runs bit for bit, the
sequence counter wrapping around."""
import ctypes

import pytest
import torch
import torch.distributed as dist

from tests.multi_gpu8 import run_on_8_gpus

TOKENS = ([1, 12, 16, 17, 33], [64, 100, 128, 129, 200], [255, 288, 300, 512])


def _worker(rank, errors, tokens):
    from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192 import DcpPackedCombine, dcp8_lse_combine_packed
    from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192.reference import combine_reference

    from .common import exact_combine, exact_name, make_sends

    dev = torch.device("cuda", rank)
    comm = DcpPackedCombine(rank, dist.group.WORLD, dev)

    def check(tag, got, sends):
        want = combine_reference(sends, rank).float()
        g = got.float()
        err = (g - want).abs().max().item()
        scale = want.abs().max().item() + 1e-6
        if not (g == g).all() or err > scale * 2**-7:
            errors.append(f"rank {rank} {tag}: max err {err:.4g} (scale {scale:.4g})")
        v = exact_combine(sends, rank)
        n = (got.view(torch.int16) != v.view(torch.int16)).sum().item()
        if n:
            errors.append(f"rank {rank} {tag}: {n} of {v.numel()} values differ from {exact_name()}")

    for t in tokens:
        for step, empty in ((0, ()), (1, (3,))):
            sends = make_sends(t, step, dev, empty)
            out = torch.full((t, 16, 192), float("nan"), dtype=torch.bfloat16, device=dev)
            dcp8_lse_combine_packed(comm, sends[rank], out)
            torch.cuda.synchronize()
            check(f"t{t} step{step} eager", out, sends)
        cap = sends[rank].clone()
        out.fill_(float("nan"))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dcp8_lse_combine_packed(comm, cap, out)
        for step in range(2, 5):
            sends = make_sends(t, step, dev)
            cap.copy_(sends[rank])
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check(f"t{t} replay{step}", out, sends)
            first = out.clone()
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(first.view(torch.int16), out.view(torch.int16)):
                errors.append(f"rank {rank} t{t} replay{step}: two runs differ")
        del graph
        if t > 1:
            # A smaller row count after a larger one: stale rows of the pools must not leak.
            sends = make_sends(t // 2, 7, dev)
            out = torch.full((t // 2, 16, 192), float("nan"), dtype=torch.bfloat16, device=dev)
            dcp8_lse_combine_packed(comm, sends[rank], out)
            torch.cuda.synchronize()
            check(f"t{t // 2} after t{t}", out, sends)


@pytest.mark.parametrize("tokens", TOKENS)
def test_combine(tokens):
    run_on_8_gpus(_worker, tokens)


def _ctl(comm, words=None):
    """This rank's control block as int32 words (``words`` written first)."""
    from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192.kernel import CTL_BYTES
    from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192.reference import WORLD
    from amd_kernels.common.hip_runtime import load_hip_runtime

    hip = load_hip_runtime()
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    ptr = int(comm.table[WORLD + comm.rank].item())
    buf = (ctypes.c_int32 * (CTL_BYTES // 4))()
    torch.cuda.synchronize()
    if words is not None:
        buf[:] = words
        assert hip.hipMemcpy(ptr, ctypes.addressof(buf), CTL_BYTES, 1) == 0
    assert hip.hipMemcpy(ctypes.addressof(buf), ptr, CTL_BYTES, 2) == 0
    return list(buf)


def _wrap_worker(rank, errors):
    from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192 import DcpPackedCombine, dcp8_lse_combine_packed
    from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192.kernel import (
        FLAG_WORD, GROUPS, MAX_TILES, SEQ_WORD, START_WORD,
    )
    from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192.reference import WORLD

    from .common import exact_combine, exact_name, make_sends

    dev = torch.device("cuda", rank)
    comm = DcpPackedCombine(rank, dist.group.WORLD, dev)
    flags = range(FLAG_WORD, FLAG_WORD + WORLD * MAX_TILES)
    starts = range(START_WORD, START_WORD + (WORLD + GROUPS) * MAX_TILES)
    for seq in (2**31 - 4, -4):  # across 2^31 - 1 -> -2^31 and -1 -> 0
        words = _ctl(comm)
        for i in (*flags, *starts, SEQ_WORD):
            words[i] = seq
        _ctl(comm, words)
        dist.barrier()
        for step, t in enumerate((512, 1, 17, 1, 300, 2, 512, 33)):
            sends = make_sends(t, step, dev)
            out = torch.full((t, 16, 192), float("nan"), dtype=torch.bfloat16, device=dev)
            dcp8_lse_combine_packed(comm, sends[rank], out)
            torch.cuda.synchronize()
            dist.barrier()
            seq = (seq + 1 + 2**31) % 2**32 - 2**31
            want = exact_combine(sends, rank)
            if not torch.equal(out.view(torch.int16), want.view(torch.int16)):
                errors.append(f"rank {rank} t{t} seq {seq}: differs from {exact_name()}")
            words = _ctl(comm)
            # Every flag and start word, used or not, holds this call's seq: none lags the counter.
            stale = [i for i in (*flags, *starts, SEQ_WORD) if words[i] != seq]
            if stale:
                errors.append(f"rank {rank} t{t} seq {seq}: {len(stale)} words not at seq, first {stale[:4]}")
            dist.barrier()  # before the peers' next call writes this rank's flags


def test_seq_wraps():
    run_on_8_gpus(_wrap_worker)
