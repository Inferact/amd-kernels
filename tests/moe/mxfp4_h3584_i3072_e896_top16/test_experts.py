# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The routed experts vs the torch reference (``reference.py``) and vs aiter's a4w4 SiTUv2 ``fused_moe``, every served M."""
import os

import pytest
import torch

os.environ.setdefault("AITER_SITUV2_A4W4", "1")

from amd_kernels.moe.mxfp4_h3584_i3072_e896_top16 import MoeWeights, moe_sort, mxfp4_moe, mxfp4_moe_experts, quantize_activations
from amd_kernels.moe.mxfp4_h3584_i3072_e896_top16 import reference as R

from .workload import SERVED_M, make_activation, make_checkpoint, make_routing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
REF_M = [1, 12, 40, 65]


@pytest.fixture(scope="module")
def ckpt():
    return make_checkpoint()


@pytest.fixture(scope="module")
def weights(ckpt):
    return MoeWeights.from_checkpoint(*ckpt)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("m", [1, 7, 12, 65, 192])
def test_quant_matches_reference(m):
    x = make_activation(m, seed=3)
    x[0, :64] = 0  # an all-zero group
    x[-1, :32] *= 1e4
    codes, scales = quantize_activations(x)
    rc, rs = R.quant_mxfp4(x.float(), floor=92)
    assert torch.equal(scales, rs)
    assert (codes != rc).float().mean().item() < 1e-4  # RNE ties only
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip
    qa, sa = per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    assert torch.equal(codes, qa.view(torch.uint8).reshape(m, -1)) and torch.equal(scales, sa.view(torch.uint8).reshape(m, -1))


@pytest.mark.parametrize("dist", ["uniform", "zipf"])
@pytest.mark.parametrize("m", REF_M)
def test_vs_reference(m, dist, ckpt, weights):
    x = make_activation(m, seed=1)
    ids, tw = make_routing(m, dist)
    out = mxfp4_moe(x, weights, ids, tw)
    ref, _ = R.moe_ref(x, ids, tw, *ckpt)
    assert rel(out, ref) < 2e-2


@pytest.mark.parametrize("m", SERVED_M)
def test_vs_aiter_and_variants(m, weights):
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType

    x = make_activation(m, seed=2)
    ids, tw = make_routing(m, "zipf")
    codes, scales = quantize_activations(x)
    sort = moe_sort(ids, tw)
    a = mxfp4_moe_experts(codes, scales, weights, sort, fused=True)
    b = mxfp4_moe_experts(codes, scales, weights, sort, fused=False)
    assert torch.equal(a, b), "one-launch and two-launch GEMMs differ"
    assert torch.equal(a, mxfp4_moe_experts(codes, scales, weights, sort, fused=True)), "two runs differ"
    fp4 = torch.float4_e2m1fn_x2
    w13 = weights.w13.view(256 * 0 + 896, 768, 1792).view(fp4)
    w2 = weights.w2.view(896, 3584, 192).view(fp4)
    w13.is_shuffled = w2.is_shuffled = True
    s13 = weights.s13.view(896, 768, 112).view(torch.float8_e8m0fnu)
    s2 = weights.s2.view(896, 3584, 16).view(torch.float8_e8m0fnu)
    stock = fm.fused_moe(x, w13, w2, tw, ids, None, ActivationType.Situv2, QuantType.per_1x32, False, s13, s2, None, None,
                         beta=4.0, linear_beta=25.0, gate_mode="separated", dtype=torch.bfloat16)
    assert rel(a, stock) < 2e-2


def test_pad_rows_and_empty_experts(weights):
    """Every token routes to the same 16 experts: 16 blocks of m rows, the rest of the 896 experts idle."""
    m = 20
    x = make_activation(m, seed=5)
    ids = torch.arange(16, dtype=torch.int32, device="cuda").repeat(m, 1)
    tw = torch.softmax(torch.randn(m, 16, device="cuda"), 1)
    out = mxfp4_moe(x, weights, ids, tw)
    assert torch.isfinite(out.float()).all()
    out2 = mxfp4_moe(x, weights, ids, tw)
    assert torch.equal(out, out2)


def test_hot_experts_many_blocks(ckpt, weights):
    """192 tokens on the same 16 experts: 12 m-blocks per expert, 880 idle experts; vs the reference."""
    m = 192
    x = make_activation(m, seed=6)
    ids = torch.arange(16, dtype=torch.int32, device="cuda").repeat(m, 1)
    ids = ids.gather(1, torch.stack([torch.randperm(16, device="cuda") for _ in range(m)]))
    tw = torch.softmax(torch.randn(m, 16, device="cuda"), 1)
    out = mxfp4_moe(x, weights, ids, tw)
    ref, _ = R.moe_ref(x, ids, tw, *ckpt)
    assert rel(out, ref) < 2e-2
    assert torch.equal(out, mxfp4_moe(x, weights, ids, tw, fused=False))


def test_slots_export(weights):
    """reduce=False: the slot rows, summed by the caller in slot order, equal the entry's output."""
    m = 40
    x = make_activation(m, seed=8)
    ids, tw = make_routing(m, "zipf")
    codes, scales = quantize_activations(x)
    sort = moe_sort(ids, tw)
    out = mxfp4_moe_experts(codes, scales, weights, sort)
    slots = mxfp4_moe_experts(codes, scales, weights, sort, reduce=False)
    rows = slots[sort.reverse_sorted.long()].view(m, 16, -1).float()
    acc = rows[:, 0].clone()
    for k in range(1, 16):
        acc = acc + rows[:, k]
    assert torch.equal(acc.to(torch.bfloat16), out)
