# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two-rank contract, bf16 quant rule, odd rows, strided residuals, both residual forms, graph overwrite, rank skew."""
import torch
import torch.distributed as dist
from amd_kernels.norm.allreduce_gemma_bf16_mxfp8_h6144_ranks2 import Bf16ArNormComm, host
from amd_kernels.norm.allreduce_gemma_bf16_mxfp8_h6144_ranks2.reference import (
    allreduce_norm_reference,
    reduce_reference,
)
from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import (
    CODE_MISMATCH,
    MXFP8_COS,
    SCALE_MISMATCH,
    compare_mxfp8,
    unshuffle_scales,
    untile_codes,
)
from tests.multi_gpu import run_on_2_gpus

# every rows-per-batch tier and form edge, and the DSpark-7 draft rows (7 per request)
ROWS = (1, 7, 17, 23, 24, 63, 64, 65, 70, 112, 128, 129, 224, 256, 257, 448, 512, 513, 896, 1025, 1792, 2048)


def bits(t):
    return t.contiguous().view(torch.uint8)


def check(got, ref, m, tile, own=None):
    """``ref``: the one-GPU bf16 norm of the reduced rows (out, residual out). bf16 out bit for bit; MXFP8 out (of the
    bf16-rounded norm) within the family reference's tolerances; the residual bit for bit."""
    if tile is None:
        assert torch.equal(bits(got[0]), bits(ref[0]))
    else:
        q = untile_codes(got[0], m) if tile else got[0][:m]
        st = compare_mxfp8(bits(q), unshuffle_scales(got[1], m), ref[0].double(), bf16_input=True)
        assert st['scale_max_diff'] <= 1 and st['scale_mismatch'] <= SCALE_MISMATCH, ('s', m, tile, st)
        assert st['code_mismatch'] <= CODE_MISMATCH and st['code_far'] == 0 and st['cos'] >= MXFP8_COS, ('q', m, tile,
                                                                                                         st)
    lo, hi = own if own else (0, m)
    assert torch.equal(bits(got[-1][lo:hi]), bits(ref[-1][lo:hi])), ('res', m, tile)


def same(got, want, m, tile, own=None):
    """Two calls' outputs bit for bit (codes and scales on the written rows; residual on ``own``)."""
    if tile is None:
        assert torch.equal(bits(got[0]), bits(want[0]))
    else:
        q = untile_codes(got[0], m) if tile else got[0][:m]
        eq = untile_codes(want[0], m) if tile else want[0][:m]
        assert torch.equal(bits(q), bits(eq)), ('q', m, tile)
        assert torch.equal(unshuffle_scales(got[1], m), unshuffle_scales(want[1], m)), ('s', m, tile)
    lo, hi = own if own else (0, m)
    assert torch.equal(bits(got[-1][lo:hi]), bits(want[-1][lo:hi])), ('res', m, tile)


def worker(rank, errors, rows=ROWS):
    dev = torch.device('cuda', rank)
    c = Bf16ArNormComm(rank, dist.group.WORLD, dev)
    w = (torch.randn(6144, generator=torch.Generator().manual_seed(72)) * 0.2).to(dev, dtype=torch.bfloat16)
    c.warm_up(w)
    from tests.mfma_spinner import mfma_spinner
    spinner = mfma_spinner(256)
    side = torch.cuda.Stream(priority=-1)
    for m in rows:
        torch.manual_seed(m)
        exponents = torch.randint(-5, 6, (m, 192, 1), device=dev).expand(m, 192, 32).reshape(m, 6144)
        parts = [(torch.randn(m, 6144, device=dev) * torch.exp2(exponents.float())).bfloat16() for _ in range(2)]
        for x in parts:
            x[0].zero_()
        parts[0][:, 64:72] = 2.0**-133  # denormal operands: flushed by the rank sum
        parts[1][:, 72:80] = -0.0
        parts[0][:, 128:160] = 2.0**-125  # normal operands that cancel into a denormal
        parts[1][:, 128:160] = -(2.0**-125) + 2.0**-133
        rbase = torch.randn((m, 6152), device=dev, dtype=torch.bfloat16)
        r = rbase[:, :6144]
        for tile in (None, 0, 16, 32, 64, 128):
            ar = reduce_reference(parts)
            y64, rr = allreduce_norm_reference(parts, r, w)
            fn = lambda: (c.ar_add_rmsnorm(parts[rank], r, w) if tile is None  # noqa: E731
                          else c.ar_add_rmsnorm_mxfp8(parts[rank], r, w, 1e-6, tile))
            ref = an.gemma_add_rmsnorm(ar, r, w, 1e-6)
            eager = fn()
            torch.cuda.synchronize()
            check(eager, ref, m, tile)
            eager = [t.clone() for t in eager]
            if tile is None:
                cos = torch.nn.functional.cosine_similarity(eager[0].double(), y64, dim=1)
                assert cos.min() > 0.99999, (m, float(cos.min()))
            g = torch.cuda.CUDAGraph()
            with c.capture(), torch.cuda.graph(g):
                out = fn()
            for rep in range(3):
                for t in out:
                    t.view(torch.uint8).fill_(255)
                if rep == 2 and tile == 16 and m in (7, 224, 896, 1792):
                    spinner(100_000, side)
                g.replay()
                torch.cuda.synchronize()
                same(out, eager, m, tile)
            # changed graph inputs must be consumed; every rank regenerates the same sign flip for its oracle
            for x in parts:
                x.neg_()
            ar2 = reduce_reference(parts)
            ref2 = an.gemma_add_rmsnorm(ar2, r, w, 1e-6)
            g.replay()
            torch.cuda.synchronize()
            check(out, ref2, m, tile)
            full = [t.clone() for t in out]
            # the owned-residual form: the same outputs, residual on the owned rows only
            own = c.owned_rows(m)
            fo = lambda: (c.ar_add_rmsnorm(parts[rank], r, w, owned_residual=True) if tile is None  # noqa: E731
                          else c.ar_add_rmsnorm_mxfp8(parts[rank], r, w, 1e-6, tile, owned_residual=True))
            same(fo(), full, m, tile, own)
            go = torch.cuda.CUDAGraph()
            with c.capture(), torch.cuda.graph(go):
                oo = fo()
            for t in oo:
                t.view(torch.uint8).fill_(255)
            go.replay()
            torch.cuda.synchronize()
            same(oo, full, m, tile, own)
        if rank == 0:
            print('contract PASS rows', m, flush=True)
    c.close()


def test_allreduce_contract():
    run_on_2_gpus(worker, timeout=900)


def worker_owned_chain(rank, errors):
    """The draft's chain: owned-residual calls feed each other's residual (only the owned rows are valid between)."""
    dev = torch.device('cuda', rank)
    c = Bf16ArNormComm(rank, dist.group.WORLD, dev)
    w = (torch.randn(6144, generator=torch.Generator().manual_seed(3)) * 0.2).to(dev, dtype=torch.bfloat16)
    c.warm_up(w)
    for m in (7, 21, 28, 63, 64, 224, 448, 1792):
        g = torch.Generator(device=dev).manual_seed(m)
        resid_ref = torch.randn(m, 6144, generator=g, device=dev).bfloat16()
        resid = resid_ref.clone()
        lo, hi = c.owned_rows(m) if m >= host.OWNER_MIN_ROWS else (0, m)
        for step in range(10):
            parts = [torch.randn(m, 6144, generator=g, device=dev).bfloat16() for _ in range(2)]
            tile = None if step == 9 else (16, 0)[step % 2]
            ref = an.gemma_add_rmsnorm(reduce_reference(parts), resid_ref, w, 1e-6)
            if tile is None:
                got = c.ar_add_rmsnorm(parts[rank], resid, w, owned_residual=True)
            else:
                got = c.ar_add_rmsnorm_mxfp8(parts[rank], resid, w, 1e-6, tile, owned_residual=True)
            torch.cuda.synchronize()
            check(got, ref, m, tile, (lo, hi))
            resid_ref, resid = ref[-1], got[-1]
    c.close()


def test_owned_residual_chain():
    run_on_2_gpus(worker_owned_chain, timeout=600)


def worker_async(rank, errors):
    import time
    dev = torch.device('cuda', rank)
    c = Bf16ArNormComm(rank, dist.group.WORLD, dev)
    w = torch.ones(6144, dtype=torch.bfloat16, device=dev)
    c.warm_up(w)
    history = []
    for call in range(36):
        m = (7, 70, 224, 448, 896, 1792, 1, 65, 513)[call % 9]
        parts = [torch.randn((m, 6144), generator=torch.Generator().manual_seed(1000 * call + r),
                             dtype=torch.bfloat16).pin_memory() for r in range(2)]
        residual = torch.zeros((m, 6144), dtype=torch.bfloat16, device=dev)
        if rank == call % 2:
            time.sleep(0.01)
        x = parts[rank].to(dev, non_blocking=True)
        tile = 16 if call % 2 else None
        owner = call % 3 == 0
        out = (c.ar_add_rmsnorm(x, residual, w, owned_residual=owner) if tile is None
               else c.ar_add_rmsnorm_mxfp8(x, residual, w, 1e-6, tile, owned_residual=owner))
        history.append((parts, residual, out, m, tile, owner))
    torch.cuda.synchronize()
    for parts, residual, out, m, tile, owner in history:
        reduced = reduce_reference([p.to(dev) for p in parts])
        ref = an.gemma_add_rmsnorm(reduced, residual, w, 1e-6)
        check(out, ref, m, tile, c.owned_rows(m) if owner else None)
    c.close()


def test_eager_rank_skew_and_slot_reuse():
    run_on_2_gpus(worker_async, timeout=300)
