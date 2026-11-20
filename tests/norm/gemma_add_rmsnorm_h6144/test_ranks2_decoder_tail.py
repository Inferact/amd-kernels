# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``allreduce_add_rmsnorm_ranks2.DecoderArNorm``: ``add_rmsnorm`` writes the bytes of the communicator call it routes
to, and ``post_attention`` / ``pre_attention`` write the bytes of the per-kernel chain they run (the norm, the MoE
entry ``moe.mxfp4_h6144_i1536_e128_top4`` or the dense MLP of ``decoder.dense_mxfp8_h6144_i6144_ranks2``, the next
norm), on 2 gfx950 GPUs; ``moe=`` warms the MoE modes the tail runs. Without the dense package a dense layer's tail
returns gate_up's codes (checked) and the dense chain case skips."""

import importlib.util

import pytest
import torch
import torch.distributed as dist

from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm
from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm_ranks2 as arn
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import H, unshuffle_scales, untile_codes

from tests.multi_gpu import run_on_2_gpus

EPS = 1e-6
I_DENSE = 6144
QKV_N = 4992
DENSE_PKG = "amd_kernels.decoder.dense_mxfp8_h6144_i6144_ranks2"
ROUTE_ROWS = (1, 8, 48, 49, 256, 1000, 1752, 4095)
MOE_ROWS = (1, 8, 64, 300, 2049, 4095)
FINAL_ROWS = (8, 300)


def _codes(q, rows, gemm_weight):
    tile = add_rmsnorm.code_tile_rows_for(rows, gemm_weight)
    return untile_codes(q, rows) if tile else q.view(torch.uint8)[:rows]


def _same_codes(a, b, rows, gemm_weight):
    (q, s), (wq, ws) = a, b
    return torch.equal(_codes(q, rows, gemm_weight), _codes(wq, rows, gemm_weight)) and torch.equal(
        unshuffle_scales(s, rows), unshuffle_scales(ws, rows))


def _route_worker(rank, errors):
    dev = torch.device("cuda", rank)
    g = torch.Generator(device=dev).manual_seed(7)
    weight = (torch.randn(H, generator=g, device=dev) * 0.1 + 1).bfloat16()
    comms = arn.DecoderArNorm(rank, dist.group.WORLD, dev, weight, EPS)
    gemm_weight = torch.empty((QKV_N, H), dtype=torch.float8_e4m3fn, device=dev)
    for rows in ROUTE_ROWS:
        g = torch.Generator(device=dev).manual_seed(rows * 10 + rank)
        x = torch.randn(rows, H, generator=g, device=dev).bfloat16()
        slots = torch.randn(rows, 4, H, generator=g, device=dev).bfloat16()
        shared = torch.randn(rows, H, generator=g, device=dev).bfloat16()
        res = torch.randn(rows, H, generator=g, device=dev).bfloat16()
        out, r = comms.add_rmsnorm(res.clone(), weight, EPS, partial=x)
        want, wr = comms.staged.ar_add_rmsnorm(x, res.clone(), weight, EPS)
        if not (torch.equal(out, want) and torch.equal(r, wr)):
            errors.append(f"rank {rank} rows {rows}: bf16 out differs")
        for name, kw, direct in (
            ("staged", dict(partial=x),
             lambda: comms.staged.ar_add_rmsnorm_mxfp8(x, res.clone(), weight, EPS, gemm_weight=gemm_weight)),
            ("slots4", dict(slots=slots, shared=shared),
             lambda: comms.slots.slots4_ar_add_rmsnorm_mxfp8(
                 slots, shared, res.clone(), weight, EPS, gemm_weight=gemm_weight)),
        ):
            (q, s), r = comms.add_rmsnorm(res.clone(), weight, EPS, gemm_weight=gemm_weight, **kw)
            wq, ws, wr = direct()
            if not (_same_codes((q, s), (wq, ws), rows, gemm_weight) and torch.equal(r, wr)):
                errors.append(f"rank {rank} rows {rows}: {name} MXFP8 out differs")


def test_ranks2_decoder_ar_norm_routes():
    run_on_2_gpus(_route_worker)


def _moe(rank, g, full=False):
    """The rank's MoE block; ``full``: the full shared expert copy (I 3072), the rank's shard computed by view."""
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import MoeWeights, SharedWeights
    from tests.moe.mxfp4_h6144_i1536_e128_top4.workload import make_experts

    raw, _ = make_experts(seed=rank)
    gate = torch.randn((128, H), generator=g, device="cuda") / H**0.5
    bias = torch.randn((128,), generator=g, device="cuda") * 0.01
    i = 3072 if full else 1536
    sq = (torch.randn((2 * i, H), device="cuda", generator=g) * 20).to(torch.float8_e4m3fn).view(torch.uint8)
    ss = torch.randint(117, 122, (2 * i, H // 32), device="cuda", dtype=torch.uint8, generator=g)
    dq = (torch.randn((H, i), device="cuda", generator=g) * 20).to(torch.float8_e4m3fn).view(torch.uint8)
    ds = torch.randint(117, 122, (H, i // 32), device="cuda", dtype=torch.uint8, generator=g)
    shared = SharedWeights.from_row_major(sq, ss, dq, ds, tp_rank=rank if full else None)
    return MoeWeights.from_row_major(gate, bias, *raw, shared=shared)


def _tail_worker(rank, errors, dense, full=False):
    from amd_kernels.moe import mxfp4_h6144_i1536_e128_top4 as moe_pkg
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import mxfp4_moe

    dev = torch.device("cuda", rank)
    g = torch.Generator(device=dev).manual_seed(11 + rank)
    norm_w = (torch.randn(H, generator=g, device=dev) * 0.1 + 1).bfloat16()
    moe = _moe(rank, g, full)
    warmed = []
    real = moe_pkg.warm_up_mxfp4_moe
    moe_pkg.warm_up_mxfp4_moe = lambda w, **kw: warmed.append((w, kw))
    try:
        comms = arn.DecoderArNorm(rank, dist.group.WORLD, dev, norm_w, EPS, moe=moe)
    finally:
        moe_pkg.warm_up_mxfp4_moe = real
    if len(warmed) != 1 or warmed[0][0] is not moe or warmed[0][1] != {"partials": True}:
        errors.append(f"rank {rank}: moe= warmed {[kw for _, kw in warmed]}")
    qkv = torch.empty((QKV_N, H), dtype=torch.float8_e4m3fn, device=dev)

    def inputs(rows):
        gi = torch.Generator(device=dev).manual_seed(rows * 10 + rank)
        return tuple(torch.randn(rows, H, generator=gi, device=dev).bfloat16() for _ in range(3))

    def check(name, rows, ok):
        if not ok:
            errors.append(f"rank {rank} rows {rows}: {name} differs")

    if dense:
        from amd_kernels.decoder import dense_mxfp8_h6144_i6144_ranks2 as D
        from amd_kernels.gemm.mxfp8_common import gemm as G

        def e4m3(n, k, scale):
            q = (torch.randn((n, k), device="cuda", generator=g) * scale).to(torch.float8_e4m3fn)
            return q, torch.randint(117, 122, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)

        mlp = arn.DenseMlp(*G.preshuffle_mxfp8_weight(*e4m3(2 * I_DENSE, H, 0.6)),
                           *G.preshuffle_mxfp8_weight(*e4m3(H, I_DENSE, 0.6)), 1.702, 1.0, 7.0)
        for rows in (1, 8, 513, D.MAX_ROWS, D.MAX_ROWS + 1):
            o, res, res2 = inputs(rows)
            x, r = comms.post_attention(o.clone(), res.clone(), norm_w, EPS, mlp)
            (q, s), wr = comms.add_rmsnorm(res.clone(), norm_w, EPS, partial=o.clone(), gemm_weight=mlp.gate_up)
            check("dense residual", rows, torch.equal(r, wr))
            if rows > D.MAX_ROWS:
                check("dense codes", rows, _same_codes(x, (q, s), rows, mlp.gate_up))
                continue
            want = D.mxfp8_swiglu_oai_mlp(q, s, rows, mlp.gate_up, mlp.gate_up_scale, mlp.down, mlp.down_scale,
                                          mlp.alpha, mlp.beta, mlp.limit, form="fp32")
            check("dense partial", rows, isinstance(x, arn.FfnPartial) and torch.equal(x.partial, want))
            got, r2 = comms.pre_attention(x, res2.clone(), norm_w, EPS, gemm_weight=qkv)
            exp, wr2 = comms.add_rmsnorm(res2.clone(), norm_w, EPS, partial=want, gemm_weight=qkv)
            check("dense next norm", rows, _same_codes(got, exp, rows, qkv) and torch.equal(r2, wr2))
        return
    if arn._dense_package() is None:
        # without the dense MLP package a dense layer's tail returns gate_up's codes for the caller's MLP
        gate_up = torch.zeros((2 * I_DENSE, H), dtype=torch.float8_e4m3fn, device=dev)
        down = torch.zeros((H, I_DENSE), dtype=torch.float8_e4m3fn, device=dev)
        mlp = arn.DenseMlp(gate_up, None, down, None, 1.702, 1.0, 7.0)
        for rows in (8, 513):
            o, res, _ = inputs(rows)
            x, r = comms.post_attention(o.clone(), res.clone(), norm_w, EPS, mlp)
            want, wr = comms.add_rmsnorm(res.clone(), norm_w, EPS, partial=o.clone(), gemm_weight=gate_up)
            check("dense codes without the MLP package", rows,
                  isinstance(x, tuple) and _same_codes(x, want, rows, gate_up) and torch.equal(r, wr))
    for final in (False, True):
        for rows in FINAL_ROWS if final else MOE_ROWS:
            o, res, res2 = inputs(rows)
            x, r = comms.post_attention(o.clone(), res.clone(), norm_w, EPS, moe, final=final)
            xb, wr = comms.add_rmsnorm(res.clone(), norm_w, EPS, partial=o.clone())
            check("moe residual", rows, torch.equal(r, wr))
            if final:
                want = mxfp4_moe(xb, moe)
                check("moe final partial", rows, x.slots is None and torch.equal(x.partial, want))
                got, r2 = comms.pre_attention(x, res2.clone(), norm_w, EPS)
                exp, wr2 = comms.add_rmsnorm(res2.clone(), norm_w, EPS, partial=want)
                check("final norm", rows, torch.equal(got, exp) and torch.equal(r2, wr2))
                continue
            slots, shared = mxfp4_moe(xb, moe, return_partials=True)
            check("moe slots", rows, torch.equal(x.slots, slots) and torch.equal(x.shared, shared))
            got, r2 = comms.pre_attention(x, res2.clone(), norm_w, EPS, gemm_weight=qkv)
            exp, wr2 = comms.add_rmsnorm(res2.clone(), norm_w, EPS, slots=slots, shared=shared, gemm_weight=qkv)
            check("moe next norm", rows, _same_codes(got, exp, rows, qkv) and torch.equal(r2, wr2))


def test_ranks2_decoder_tail_moe_matches_the_chain():
    run_on_2_gpus(_tail_worker, False, timeout=2400)  # the MoE entry compiles each row count on its first call here


def test_ranks2_decoder_tail_moe_full_shared_copy():
    """The decode tail on the full shared expert copy (each rank's TP shard read by view)."""
    run_on_2_gpus(_tail_worker, False, True, timeout=2400)


@pytest.mark.skipif(importlib.util.find_spec(DENSE_PKG) is None, reason=f"needs {DENSE_PKG}")
def test_ranks2_decoder_tail_dense_matches_the_chain():
    run_on_2_gpus(_tail_worker, True, timeout=2400)
