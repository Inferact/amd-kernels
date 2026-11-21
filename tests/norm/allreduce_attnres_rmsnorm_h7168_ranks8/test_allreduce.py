# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank contract at the served row counts: eager and graph calls, replayed inputs, bank write, determinism."""
import torch
import torch.distributed as dist
from amd_kernels.norm.allreduce_attnres_rmsnorm_h7168_ranks8 import AttnResComm
from amd_kernels.norm.allreduce_attnres_rmsnorm_h7168_ranks8.reference import (
    allreduce_attnres_rmsnorm_reference,
    unpack_tiles,
)

from tests.norm.allreduce_attnres_rmsnorm_h7168_ranks8.multi_gpu8 import run_on_8_gpus
from tests.norm.allreduce_attnres_rmsnorm_h7168_ranks8.workload import CASES, SERVED_ROWS, make


def bits(t):
    return t.contiguous().view(torch.uint8)


def check(out, pout, bank, parts, prefix, bank0, nw, qk, ow, nb, widx, on, tag):
    y, u, bank_ref = allreduce_attnres_rmsnorm_reference(
        parts, prefix, bank0, nw, qk, nb, widx, ow if on else None)
    assert torch.equal(bits(pout), bits(u)), ("prefix out", tag)
    assert torch.equal(bits(bank), bits(bank_ref)), ("bank", tag)
    o = out.double()
    err = (o - y).abs()
    tol = 2.0**-7 * y.abs() + 1e-5 * y.abs().amax(-1, keepdim=True)
    assert bool((err <= tol).all()), ("out", tag, float(err.max()), float((err / tol).max()))
    cos = torch.nn.functional.cosine_similarity(o, y, dim=1)
    assert float(cos.min()) >= 0.99999, ("cos", tag, float(cos.min()))


def worker(rank, errors):
    dev = torch.device("cuda", rank)
    c = AttnResComm(rank, dist.group.WORLD, dev)
    for m in SERVED_ROWS:
        for ci, (hp, nb, widx, on) in enumerate(CASES):
            seed = m * 10 + ci
            parts, prefix0, bank0, nw, qk, ow = make(m, seed, dev)

            def call(x, prefix, bank):
                return c.allreduce_attnres_rmsnorm(x, prefix, bank, nw, qk, nb, widx, ow if on else None)

            prefix, bank = prefix0.clone(), bank0.clone()
            out, pout = call(parts[rank], prefix if hp else None, bank)
            torch.cuda.synchronize()
            tag = (m, hp, nb, widx, on)
            check(out, pout, bank, parts, prefix0 if hp else None, bank0, nw, qk, ow, nb, widx, on, ("eager",) + tag)
            first = out.clone()
            g = torch.cuda.CUDAGraph()
            prefix, bank = prefix0.clone(), bank0.clone()
            with torch.cuda.graph(g):
                gout, gpout = call(parts[rank], prefix if hp else None, bank)
            for it in range(3):
                prefix.copy_(prefix0)
                bank.copy_(bank0)
                gout.view(torch.uint8).fill_(255)
                g.replay()
                torch.cuda.synchronize()
                assert torch.equal(bits(gout), bits(first)), ("replay", it) + tag
                check(gout, gpout, bank, parts, prefix0 if hp else None, bank0, nw, qk, ow, nb, widx, on,
                      ("graph",) + tag)
            if ci == 0:
                pk, _ = c.allreduce_attnres_rmsnorm(parts[rank], prefix0.clone(), bank0.clone(), nw, qk, nb, widx, ow,
                                                    out_packed=True)
                torch.cuda.synchronize()
                assert torch.equal(bits(unpack_tiles(pk, m)), bits(first)), ("packed", m)
            parts[rank].neg_()
            for x in parts:
                if x is not parts[rank]:
                    x.neg_()
            prefix.copy_(prefix0)
            bank.copy_(bank0)
            g.replay()
            torch.cuda.synchronize()
            check(gout, gpout, bank, parts, prefix0 if hp else None, bank0, nw, qk, ow, nb, widx, on,
                  ("changed",) + tag)
        if rank == 0:
            print("contract PASS rows", m, flush=True)
    c.close()


def test_allreduce_contract():
    run_on_8_gpus(worker, timeout=600)


def worker_skew(rank, errors):
    """Calls queued back to back with a different rank delayed before each, rows changing: the parity sets and the
    per-row counters stay in step; every output is checked after the whole sequence."""
    dev = torch.device("cuda", rank)
    c = AttnResComm(rank, dist.group.WORLD, dev)
    history = []
    rows = (12, 1, 65, 192, 24, 128, 40, 80)
    for call in range(48):
        m = rows[call % len(rows)]
        hp, nb, widx, on = CASES[call % len(CASES)]
        parts, prefix0, bank0, nw, qk, ow = make(m, 5000 + call, dev)
        prefix, bank = prefix0.clone(), bank0.clone()
        if rank == call % 8:
            torch.cuda._sleep(3_000_000)
        out, pout = c.allreduce_attnres_rmsnorm(parts[rank], prefix if hp else None, bank, nw, qk, nb, widx,
                                                ow if on else None)
        history.append((out, pout, bank, parts, prefix0 if hp else None, bank0, nw, qk, ow, nb, widx, on, m))
    torch.cuda.synchronize()
    for i, (out, pout, bank, parts, prefix0, bank0, nw, qk, ow, nb, widx, on, m) in enumerate(history):
        check(out, pout, bank, parts, prefix0, bank0, nw, qk, ow, nb, widx, on, ("skew", i, m))
    c.close()


def test_rank_skew_and_row_changes():
    run_on_8_gpus(worker_skew, timeout=300)


def worker_nan(rank, errors):
    """Partials of 0xFFFF bf16 NaNs (the empty mailbox pattern) complete and give NaN; the next call is clean."""
    dev = torch.device("cuda", rank)
    c = AttnResComm(rank, dist.group.WORLD, dev)
    m = 12
    parts, prefix0, bank0, nw, qk, ow = make(m, 777, dev)
    bad = torch.full((m, 7168), 0xFFFF, dtype=torch.int32, device=dev).to(torch.int16).view(torch.bfloat16)
    if rank == 3:
        parts[3] = bad
    out, pout = c.allreduce_attnres_rmsnorm(parts[rank], prefix0.clone(), bank0.clone(), nw, qk, 3, -1, ow)
    torch.cuda.synchronize()
    if not bool(torch.isnan(out).all()):
        errors.append(f"rank {rank}: NaN partial did not propagate")
    parts[3] = make(m, 778, dev)[0][3]
    prefix, bank = prefix0.clone(), bank0.clone()
    out, pout = c.allreduce_attnres_rmsnorm(parts[rank], prefix, bank, nw, qk, 3, -1, ow)
    torch.cuda.synchronize()
    check(out, pout, bank, parts, prefix0, bank0, nw, qk, ow, 3, -1, True, ("after nan",))
    c.close()


def test_nan_partial_does_not_hang():
    run_on_8_gpus(worker_nan, timeout=120)
