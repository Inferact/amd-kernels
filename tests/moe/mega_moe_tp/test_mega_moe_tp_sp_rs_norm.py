# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""SpRsNorm (MegaMoE TP's sequence-parallel reduce-scatter + residual add + Gemma RMSNorm, + its router) on 4 ranks
vs ``reference.sp_rs_norm_ref`` / ``reference.sigmoid_topk_ref``, twice back to back on new inputs."""

import pytest
import torch
import torch.distributed as dist
from tests.multi_gpu import WORLD, run_on_4_gpus

from amd_kernels.moe.mega_moe_tp import SpRsNorm, mega_moe_tp_supported
from amd_kernels.moe.mega_moe_tp.reference import sigmoid_topk_ref, sp_rs_norm_ref

H = 6144
EPS = 1e-6
ROUTER = (128, 4, 2.0, 0.5)  # experts, top-k, routed scale, shared-expert weight
_OPS = {}


def _inputs(tokens, router, seed, rank, dev):
    """this rank's partials of every token, the residual, the norm weight (+ the router's gate, bias, outputs)"""
    E, K = ROUTER[:2]
    gen = torch.Generator(device=dev).manual_seed(7)
    wg = torch.randn((E, H), generator=gen, device=dev) * H**-0.5 * 3
    bias = (0.05 * torch.randn((E,), generator=gen, device=dev)).float()
    w = (0.1 * torch.randn((H,), generator=gen, device=dev)).to(torch.bfloat16)
    g = torch.Generator(device=dev).manual_seed(seed + rank + tokens)
    part = torch.randn((tokens, H), generator=g, device=dev).to(torch.bfloat16)
    res = torch.randn((tokens, H), generator=g, device=dev).to(torch.bfloat16)
    rt = None
    if router:
        m = tokens // WORLD
        ids = torch.empty((m, K + 1), dtype=torch.int32, device=dev)
        tw = torch.empty((m, K + 1), dtype=torch.float32, device=dev)
        rt = (wg.to(torch.bfloat16), bias, ids, tw)
    return part, res, w, rt


def _worker(rank, errors, tokens, router):
    dev = torch.device("cuda", rank)
    if "op" not in _OPS:
        _OPS["op"] = SpRsNorm(H, 1024, EPS, device=dev, router=ROUTER)
    op = _OPS["op"]
    if router and not op.routes(tokens):
        errors.append(f"the router does not run at {tokens} tokens")
        return
    E, K, scale, shared_w = ROUTER
    m = tokens // WORLD
    rows = slice(rank * m, (rank + 1) * m)
    for seed in (1000, 2000):
        part, res, w, rt = _inputs(tokens, router, seed, rank, dev)
        out, res_out = op(part, res, w, router=rt)
        torch.cuda.synchronize()
        tot = part.float().cpu()
        dist.all_reduce(tot)
        h, ref = sp_rs_norm_ref(tot[rows].to(dev), res[rows], w, EPS)
        e = max(float((res_out[rows].float() - h).norm() / h.norm()),
                float((out[rows].float() - ref).norm() / ref.norm()))
        if op.poll_errors() or not e < 0.01:
            errors.append(f"rank {rank} M={tokens} router={router}: rel_l2 {e:.4f}, watchdog {op.poll_errors()}")
        if rt is None:
            continue
        _, _, ids, tw = rt
        rtw, rid = sigmoid_topk_ref(ref.to(torch.bfloat16), rt[0], rt[1], K, scale)
        same = (ids[:, :K].sort(1).values == rid.sort(1).values).all(1)
        mine = torch.gather(tw[:, :K], 1, ids[:, :K].argsort(1))
        want = torch.gather(rtw, 1, rid.argsort(1))
        e_w = ((mine - want).abs() / want.abs().clamp_min(1e-6))[same]
        e_w = float(e_w.max()) if e_w.numel() else 0.0
        # bf16 logits: a near tie may pick another expert (the share is over every rank's rows)
        st = torch.tensor([float(same.sum()), m])
        dist.all_reduce(st)
        frac = float(st[0] / st[1])
        if frac < 0.9 or e_w > 0.02 or not bool((ids[:, K] == E).all() and (tw[:, K] == shared_w).all()):
            errors.append(f"rank {rank} M={tokens} router: same experts {frac:.3f}, weight err {e_w:.1e}")


@pytest.mark.parametrize(("tokens", "router"), [(64, True), (1024, True), (64, False), (1000, False)])
def test_mega_moe_tp_sp_rs_norm(tokens, router):
    if not torch.cuda.is_available() or not mega_moe_tp_supported():
        pytest.skip("needs gfx950")
    run_on_4_gpus(_worker, tokens, router)
