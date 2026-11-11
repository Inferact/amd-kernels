# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""NV's Q8KV4 numerics as a torch reference (``reference.nv_sparse_prefill_reference`` and the decode entry's
``nv_sparse_decode_reference``, the fp8 target): one rule serves both phases (the two references agree bit for bit on
the same tokens), ``query.pack_query`` is NV's Q quant, and the fp8 kernel (bf16 output) is near the rule."""

import pytest
import torch

from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import reference as dref
from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import (
    nvfp4_sparse_prefill,
    reference as ref,
)
from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.query import (
    FP8_Q_ORDER,
    pack_query,
)

from .cases import SM_SCALE, TP4_HKV, make_case, require_gfx950

# the fp8 kernel rounds P against its chunk's max, the rule against the running max in slot order: rel L2 7.9e-3 -
# 1.6e-2 measured at 1 KV head (MI355X); MSA's own kernel tests allow atol / rtol 2e-2 - 3e-2
NV_REL_L2 = 2.5e-2


def test_pack_query_is_nv_q8():
    q = (torch.randn(64, 16, 128, generator=torch.Generator().manual_seed(0)) * 30).bfloat16()
    q[0, 0, :4] = torch.tensor([500.0, -1000.0, 1e-9, 0.0]).bfloat16()  # saturation, tiny
    codes = ref.quantize_query_e4m3(q).view(torch.uint8)
    packed = pack_query(q).reshape(-1, 8)
    order = torch.tensor(FP8_Q_ORDER)
    assert torch.equal(packed[:, torch.argsort(order)].reshape(q.shape), codes.reshape(q.shape))


@pytest.mark.parametrize("hkv", [TP4_HKV])
def test_decode_and_prefill_rules_agree(hkv):
    """The same tokens as prefill rows and as decode rows (one request each, ql 1)."""
    require_gfx950()
    case = make_case([300, 41], [2000, 700], hkv, seed=7, picks="shared")
    rows = torch.tensor([0, 1, 127, 128, 299, 300, 320, 340], device="cuda")
    pre = ref.nv_sparse_prefill_reference(
        case.q, case.cache, case.topk, case.table, case.req, case.pos, hkv, SM_SCALE, rows=rows
    )
    dec = dref.nv_sparse_decode_reference(
        case.q[rows],
        case.cache,
        case.topk[:, rows].contiguous(),
        case.table[case.req[rows].long()],
        (case.pos[rows] + 1).int(),
        SM_SCALE,
        1,
    )
    assert torch.equal(pre, dec)


@pytest.mark.parametrize("hkv", [TP4_HKV])
def test_archived_fp8_kernel_near_nv_rule(hkv):
    require_gfx950()
    case = make_case([1027, 514, 3, 262], [3000, 514, 1100, 900], hkv, seed=1, picks="shared")
    out = torch.empty_like(case.q)
    nvfp4_sparse_prefill(*case.args(pack_query(case.q)), out, compute_dtype="fp8")
    rows = case.rows()
    nv = ref.nv_sparse_prefill_reference(
        case.q, case.cache, case.topk, case.table, case.req, case.pos, hkv, SM_SCALE, rows=rows
    ).double()
    assert torch.isfinite(out[rows]).all()
    err = ref.rel_l2(out[rows], nv)
    print(f"fp8 kernel vs NV rule, {hkv} KV heads: rel L2 {err:.3e}, cos {ref.cosine(out[rows], nv):.7f}")
    assert err < NV_REL_L2, err
    assert ref.cosine(out[rows], nv) > 0.9995
