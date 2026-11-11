# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``DecoderArNorm.post_attention`` / ``pre_attention`` write the bytes of the per-kernel chain they run (the norm, the
MoE entry or the dense MLP entry, the next norm) below and above the dense MLP's 2048 rows and the communicators'
4095, on 4 gfx950 GPUs; ``moe=`` warms the MoE modes the tail runs."""

import torch
import torch.distributed as dist

from amd_kernels.decoder.dense_mxfp8_h6144_i3072_ranks4 import mxfp8_swiglu_oai_mlp
from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.moe import mxfp4_h6144_i768_e128_top4 as moe_pkg
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeWeights, SharedWeights, mxfp4_moe
from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm
from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm as arn
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import H, unshuffle_scales, untile_codes

from tests.moe.mxfp4_h6144_i768_e128_top4.workload import make_experts
from tests.multi_gpu import run_on_4_gpus

EPS = 1e-6
I_DENSE = 3072
DENSE_ROWS = (1, 8, 513, 2048, 2049, 4095)  # the MLP to 2048 rows, above it the codes come back
MOE_ROWS = (1, 8, 300, 2049, 4095)
FINAL_ROWS = (8, 300)


def _codes(q, rows, gemm_weight):
    tile = add_rmsnorm.code_tile_rows_for(rows, gemm_weight)
    return untile_codes(q, rows) if tile else q.view(torch.uint8)[:rows]


def _same_codes(a, b, rows, gemm_weight):
    (q, s), (wq, ws) = a, b
    return torch.equal(_codes(q, rows, gemm_weight), _codes(wq, rows, gemm_weight)) and torch.equal(
        unshuffle_scales(s, rows), unshuffle_scales(ws, rows))


def _weights(rank, g):
    raw, _ = make_experts(seed=rank)
    gate = torch.randn((128, H), generator=g, device="cuda") / H**0.5
    bias = torch.randn((128,), generator=g, device="cuda") * 0.01

    def e4m3(n, k, scale):
        q = (torch.randn((n, k), device="cuda", generator=g) * scale).to(torch.float8_e4m3fn)
        return q, torch.randint(117, 122, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)

    def u8(q, s):
        return q.view(torch.uint8), s

    shared = SharedWeights.from_row_major(*u8(*e4m3(1536, H, 20)), *u8(*e4m3(H, 768, 20)))
    moe = MoeWeights.from_row_major(gate, bias, *raw, shared=shared)
    gu = G.preshuffle_mxfp8_weight(*e4m3(2 * I_DENSE, H, 0.6))
    dw = G.preshuffle_mxfp8_weight(*e4m3(H, I_DENSE, 0.6))
    return moe, arn.DenseMlp(*gu, *dw, 1.702, 1.0, 7.0)


def _worker(rank, errors):
    dev = torch.device("cuda", rank)
    g = torch.Generator(device=dev).manual_seed(11 + rank)
    norm_w = (torch.randn(H, generator=g, device=dev) * 0.1 + 1).bfloat16()
    moe, mlp = _weights(rank, g)
    warmed = []
    real = moe_pkg.warm_up_mxfp4_moe
    moe_pkg.warm_up_mxfp4_moe = lambda w, **kw: warmed.append((w, kw))
    try:
        comms = arn.DecoderArNorm(rank, dist.group.WORLD, dev, norm_w, EPS, moe=moe)
    finally:
        moe_pkg.warm_up_mxfp4_moe = real
    if len(warmed) != 1 or warmed[0][0] is not moe or warmed[0][1] != {"partials": True}:
        errors.append(f"rank {rank}: moe= warmed {[kw for _, kw in warmed]}")
    qkv = torch.empty((2560, H), dtype=torch.float8_e4m3fn, device=dev)

    def inputs(rows):
        gi = torch.Generator(device=dev).manual_seed(rows * 10 + rank)
        return tuple(torch.randn(rows, H, generator=gi, device=dev).bfloat16() for _ in range(3))

    def check(name, rows, ok):
        if not ok:
            errors.append(f"rank {rank} rows {rows}: {name} differs")

    for rows in DENSE_ROWS:
        o, res, res2 = inputs(rows)
        x, r = comms.post_attention(o.clone(), res.clone(), norm_w, EPS, mlp)
        (q, s), wr = comms.add_rmsnorm(res.clone(), norm_w, EPS, partial=o.clone(), gemm_weight=mlp.gate_up)
        check("dense residual", rows, torch.equal(r, wr))
        if rows > 2048:
            check("dense codes", rows, _same_codes(x, (q, s), rows, mlp.gate_up))
            continue
        want = mxfp8_swiglu_oai_mlp(q, s, rows, mlp.gate_up, mlp.gate_up_scale, mlp.down, mlp.down_scale,
                                    mlp.alpha, mlp.beta, mlp.limit, form="fp32")
        check("dense partial", rows, isinstance(x, arn.FfnPartial) and torch.equal(x.partial, want))
        got, r2 = comms.pre_attention(x, res2.clone(), norm_w, EPS, gemm_weight=qkv)
        exp, wr2 = comms.add_rmsnorm(res2.clone(), norm_w, EPS, partial=want, gemm_weight=qkv)
        check("dense next norm", rows, _same_codes(got, exp, rows, qkv) and torch.equal(r2, wr2))
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
    o, res, _ = inputs(4096)
    for ffn in (mlp, moe):
        check("above the communicators' rows", 4096, comms.post_attention(o, res, norm_w, EPS, ffn) is None)


def test_decoder_tail_matches_the_chain():
    run_on_4_gpus(_worker, timeout=2400)  # the entries compile each row count on its first call here
