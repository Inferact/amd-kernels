# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``q_gather_comm`` / ``q_bproj_wuk_gather`` on 8 GPUs: the communicator built and the weights packed through the
entry; at the smallest and the largest served row count, for ``out``, ``out_q8`` and both, the entry's outputs equal
the shape package's direct call on the same communicator byte for byte with the reason None; another entry's
communicator gives a reason; no row count compiles after the warm-up; a graph of the entry call replays the eager
result."""

import torch
import torch.distributed as dist

from tests.multi_gpu8 import run_on_8_gpus

ROWS = (1, 256)
PROBES = (1, 16, 17, 32, 48, 64, 80, 96, 128, 160, 192, 256)
DIMS = dict(q_lora=1536, q_b_rows=2304, heads=96)


def _worker(rank, errors):
    from amd_kernels import warm_up
    from amd_kernels.attention import mla_decode_common as E
    from amd_kernels.attention.dcp8_q_bproj_wuk_gather_h96_kv512 import pack_q_b, pack_w_uk, q_b_allgather_wuk_all_heads

    from .cases import same_bytes, tensors
    from .ranks8 import graph_matches_eager, walk_without_compiles

    dev = torch.device("cuda", rank)
    warm_up.set_auto(True)
    warm_up.set_workers(4)
    group = dist.group.WORLD
    g = torch.Generator(device=dev).manual_seed(7)
    rnd = lambda *s, scale=1.0: (torch.randn(*s, device=dev, generator=g) * scale).bfloat16()  # noqa: E731
    w_qb = [rnd(2304, 1536, scale=0.03) for _ in range(8)]
    w_uk = torch.cat([rnd(12, 128, 512, scale=0.05) for _ in range(8)])
    why = E.q_gather_comm_blocker(rank, group, dev, "q", **DIMS)
    if why is not None:
        errors.append(f"rank {rank}: factory reason {why}")
    comm = E.q_gather_comm(rank, group, dev, "q", **DIMS)
    wq, wu = E.pack_q_b(w_qb[rank]), E.pack_w_uk(w_uk)
    if not (same_bytes(wq, pack_q_b(w_qb[rank])) and same_bytes(wu, pack_w_uk(w_uk))):
        errors.append(f"rank {rank}: packed weights differ")

    def buffers(m, form):
        out = torch.zeros((m, 96, 576), dtype=torch.bfloat16, device=dev) if form != "q8" else None
        q8 = torch.zeros((m, 96, 1168), dtype=torch.uint8, device=dev) if form != "bf16" else None
        return {"out": out, "out_q8": q8}

    for m in ROWS:
        q_a = rnd(m, 1536)
        for form in ("bf16", "q8", "both"):
            want, have = buffers(m, form), buffers(m, form)
            why = E.q_bproj_wuk_gather_blocker(comm, q_a, wq, wu, **have)
            if why is not None:
                errors.append(f"rank {rank} m{m} {form}: reason {why}")
            q_b_allgather_wuk_all_heads(comm, q_a, wq, wu, **want)
            E.q_bproj_wuk_gather(comm, q_a, wq, wu, **have)
            torch.cuda.synchronize()
            if not all(same_bytes(a, b) for a, b in zip(tensors(have), tensors(want))):
                errors.append(f"rank {rank} m{m} {form}: entry != package")
    lse = torch.zeros((4, 96), dtype=torch.float32, device=dev)
    x = torch.zeros((4, 96, 512), dtype=torch.bfloat16, device=dev)
    why = E.lse_combine_vup_blocker(comm, x, lse, x, x)
    if not (why or "").startswith("lse_combine_vup: no kernel for communicator QueryGatherComm"):
        errors.append(f"rank {rank}: cross-entry communicator reason {why}")

    def call(m, form):
        q_a, bufs = rnd(m, 1536), buffers(m, form)
        return lambda: E.q_bproj_wuk_gather(comm, q_a, wq, wu, **bufs)

    walk_without_compiles(rank, errors, [call(m, f) for f in ("bf16", "q8", "both") for m in PROBES], "query")
    q_a, bufs = rnd(48, 1536), buffers(48, "both")
    graph_matches_eager(rank, errors, lambda: E.q_bproj_wuk_gather(comm, q_a, wq, wu, **bufs), bufs, "query m48")
    torch.cuda.synchronize()
    comm.close()


def test_entry_on_8_gpus(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "flydsl"))  # every variant really compiles
    run_on_8_gpus(_worker, timeout=110)
