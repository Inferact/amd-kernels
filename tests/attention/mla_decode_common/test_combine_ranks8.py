# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``lse_combine_comm`` / ``lse_combine_vup`` on 8 GPUs: the communicator built through the entry; at the smallest and
the largest served row count, with and without the gate and ``seq_lens``, the entry's output equals the shape package's
direct call on the same communicator byte for byte with the reason None; another entry's communicator gives a reason;
no row count compiles after the warm-up; a graph of the entry call replays the eager result."""

import torch
import torch.distributed as dist

from tests.multi_gpu8 import run_on_8_gpus

ROWS = (1, 256)
PROBES = (1, 12, 16, 17, 33, 48, 65, 80, 128, 129, 160, 192, 193, 240, 256)
DIMS = dict(heads=96, v_head=128, kv_lora=512)


def _worker(rank, errors):
    from amd_kernels import warm_up
    from amd_kernels.attention import mla_decode_common as E
    from amd_kernels.attention.dcp8_lse_combine_vup_mla_h12_kv512 import dcp8_lse_combine_vup, prepare_w_uv

    from tests.attention.dcp8_lse_combine_vup_mla_h12_kv512.test_combine import make_inputs

    from .cases import same_bytes
    from .ranks8 import graph_matches_eager, walk_without_compiles

    dev = torch.device("cuda", rank)
    warm_up.set_auto(True)
    warm_up.set_workers(4)
    group = dist.group.WORLD
    why = E.lse_combine_comm_blocker(rank, group, dev, **DIMS)
    if why is not None:
        errors.append(f"rank {rank}: factory reason {why}")
    comm = E.lse_combine_comm(rank, group, dev, **DIMS)
    if comm.max_tokens != 256:
        errors.append(f"rank {rank}: max_tokens {comm.max_tokens}")
    for t in ROWS:
        xs, lses, w_uv, sls, qsls = make_inputs(t, 0, dev)
        wt = E.prepare_w_uv(w_uv)
        if not same_bytes(wt, prepare_w_uv(w_uv)):
            errors.append(f"rank {rank}: prepare_w_uv differs")
        gate = (torch.randn((t, 1536), device=dev) * 3).bfloat16()
        for g in (None, gate):
            for s, q in ((None, None), (sls[rank], qsls[rank])):
                outs = [torch.zeros((t, 12, 128), dtype=torch.bfloat16, device=dev) for _ in range(2)]
                args = (comm, xs[rank], lses[rank], wt, outs[1], s, q, g)
                why = E.lse_combine_vup_blocker(*args)
                if why is not None:
                    errors.append(f"rank {rank} t{t}: reason {why}")
                dcp8_lse_combine_vup(comm, xs[rank], lses[rank], wt, outs[0], s, q, g)
                got = E.lse_combine_vup(*args)
                torch.cuda.synchronize()
                if got is not outs[1] or not same_bytes(outs[0], outs[1]):
                    errors.append(f"rank {rank} t{t} gate={g is not None} seq={s is not None}: entry != package")
    q_a = torch.zeros((4, 1536), dtype=torch.bfloat16, device=dev)
    why = E.q_bproj_wuk_gather_blocker(comm, q_a, q_a, q_a)
    if not (why or "").startswith("q_bproj_wuk_gather: no kernel for communicator DcpVupCombine"):
        errors.append(f"rank {rank}: cross-entry communicator reason {why}")

    def call(t, gated):
        xs, lses, _, _, _ = make_inputs(t, 1, dev, pad_heads=128)
        out = torch.empty((t, 12, 128), dtype=torch.bfloat16, device=dev)
        g = torch.zeros((t, 1536), dtype=torch.bfloat16, device=dev) if gated else None
        return lambda: E.lse_combine_vup(comm, xs[rank], lses[rank], wt, out, None, None, g)

    walk_without_compiles(rank, errors, [call(t, gated) for gated in (False, True) for t in PROBES], "combine")
    xs, lses, _, sls, qsls = make_inputs(80, 2, dev)
    out = torch.zeros((80, 12, 128), dtype=torch.bfloat16, device=dev)
    gate = torch.randn((80, 1536), device=dev).bfloat16()
    graph_matches_eager(rank, errors, lambda: E.lse_combine_vup(comm, xs[rank], lses[rank], wt, out, sls[rank],
                                                                qsls[rank], gate), [out], "combine t80")


def test_entry_on_8_gpus(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "flydsl"))  # every variant really compiles
    run_on_8_gpus(_worker, timeout=110)
