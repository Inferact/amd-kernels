# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sort and GEMM1 + clamp-SwiGLU + MXFP8 quant of every dispatch variant against the reference intermediate."""
import pytest
import torch
from amd_kernels.moe.mxfp4_h5120_i2304_e48_top6 import ExpertWeights, host, reference as ref

from .workload import make_activation, make_experts, make_routes, require_gfx950

E, I, TOPK = ref.E, ref.I, ref.TOPK


@pytest.fixture(scope="module")
def experts():
    require_gfx950()
    raw = make_experts(seed=5)
    return raw, ExpertWeights.from_row_major(*raw)


@pytest.mark.parametrize("bm", [16, 32, 64, 128])
@pytest.mark.parametrize("n", [1, 16, 100, 768])
@torch.inference_mode()
def test_sort_places_every_pair_once(n, bm):
    require_gfx950()
    ids, wt = make_routes(n, "skew", seed=n)
    sid, wts, eids, nv = host.moe_sort(ids, wt, bm)
    torch.cuda.synchronize()
    rows = int(nv)
    counts = torch.bincount(ids.flatten().long(), minlength=E)
    assert rows == int(((counts + bm - 1) // bm * bm).sum())
    token, slot = (sid[:rows] & 0xFFFFFF).long(), (sid[:rows] >> 24).long()
    real = token < n
    assert int(real.sum()) == n * TOPK
    block_expert = eids[: rows // bm].long().repeat_interleave(bm)
    assert torch.equal(ids[token[real], slot[real]].long(), block_expert[real])
    assert torch.equal(wts[:rows][real], wt[token[real], slot[real]])
    assert (wts[:rows][~real] == 0).all()
    assert len({(int(t), int(s)) for t, s in zip(token[real], slot[real])}) == n * TOPK


CASES = sorted({(cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[9]) for _, cfg in host.DISPATCH}
               | {(16, 2, 3, 2, 0, 4), (32, 4, 3, 2, 0, 4), (32, 1, 3, 4, 0, 4), (64, 2, 2, 4, 0, 4),
                  (128, 1, 3, 2, 3, 4), (64, 1, 2, 2, 4, 8), (128, 1, 2, 2, 3, 8)})


@pytest.mark.parametrize("case", CASES, ids=str)
@torch.inference_mode()
def test_gemm1_rows(experts, case):
    raw, w = experts
    bm, kw, pf, ni, nbuf, nwv = case
    n = 192
    xc, xs = make_activation(n, seed=11)
    ids, wt = make_routes(n, "uniform", seed=11)
    sid, wts, eids, nv = host.moe_sort(ids, wt, bm)
    rows = sid.numel()
    args = dict(bm=bm, kw=kw, pf=pf, rows=rows, ni=ni, nbuf=nbuf, nwv=nwv)
    q, s = host.gemm1_swiglu_quant(xc, xs, w, sid, eids, nv, **args)
    q2, s2 = host.gemm1_swiglu_quant(xc, xs, w, sid, eids, nv, **args)
    nvalid = int(nv)
    codes, scales = ref.unpack_rows(q, s, rows)
    codes2, scales2 = ref.unpack_rows(q2, s2, rows)
    token, slot = (sid[:nvalid] & 0xFFFFFF).long(), (sid[:nvalid] >> 24).long()
    real = (token < n).nonzero().flatten()
    assert torch.equal(codes[real], codes2[real]) and torch.equal(scales[real], scales2[real])
    got = ref.dequant_mxfp8(codes[real].view(torch.float8_e4m3fn), scales[real])
    x = ref.dequant_mxfp8(xc, xs)
    cos = []
    for r in real[::7].tolist():
        e = int(ids[token[r], slot[r]])
        gu = x[token[r]] @ ref.dequant_mxfp4(raw[0][e], raw[1][e]).t()
        act = ref.dequant_mxfp8(*ref.quantize_mxfp8(ref.clamp_silu_mul(gu[:I], gu[I:])[None]))[0]
        i = int((real == r).nonzero())
        cos.append(torch.nn.functional.cosine_similarity(act, got[i], dim=0).item())
    # a row of 2304 codes can hold one code that fp32 summation order flips (cosine -3e-5)
    assert min(cos) >= 0.9999 and sum(cos) / len(cos) >= 0.999999
