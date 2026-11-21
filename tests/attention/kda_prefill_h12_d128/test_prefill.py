# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The whole prefill (conv + prologue, scan, gated norm) against reference.kda_prefill_reference, on both paths."""

import pytest
import torch

from amd_kernels.attention.kda_prefill_h12_d128 import host
from amd_kernels.attention.kda_prefill_h12_d128 import reference as R

from .inputs import make_case, require_gfx950

OUT_TOL = 8e-3
STATE_TOL = 5e-3
IMPLS = ["mega", "v1"]


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def chunk_indices(cu):
    rows = [(n, c) for n, c, _, _ in R.chunk_table(cu)[0]]
    return torch.tensor(rows, dtype=torch.int32, device=cu.device)


def offsets(cu):
    return torch.tensor(R.chunk_table(cu)[1], dtype=torch.int64, device=cu.device)


def counts(cu):
    cl = cu.tolist()
    return [-(-(b - a) // 64) for a, b in zip(cl[:-1], cl[1:])]


def run(c, impl=None, **extra):
    cu = c["cu_seqlens"]
    out = torch.zeros(int(cu[-1]), R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    conv_state, state = c["conv_state"].clone(), c["state_cache"].clone()
    host.prefill(c["mixed"], c["raw_g"], c["beta_logit"], c["gate"], c["A_log"], c["dt_bias"], c["conv_weight"], conv_state, state,
                 c["state_indices"], c["has_initial_state"], cu, chunk_indices(cu), offsets(cu), c["norm_weight"], c["norm_eps"], out,
                 c["lower_bound"], chunk_counts=counts(cu), impl=impl, **extra)
    return out, conv_state, state


def reference(c, **extra):
    conv_state, state = c["conv_state"].clone(), c["state_cache"].clone()
    out = R.kda_prefill_reference(c["mixed"], c["raw_g"], c["beta_logit"], c["gate"], c["A_log"], c["dt_bias"], c["lower_bound"],
                                  c["conv_weight"], conv_state, state, c["state_indices"], c["has_initial_state"], c["cu_seqlens"],
                                  c["norm_weight"], c["norm_eps"], **extra)
    return out, conv_state, state


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("seq,init,dt_shift", [
    ([64], [True], 0.0), ([130], [False], 0.0), ([2048], [True], 0.0), ([300, 64, 77, 1], [True, False, True, True], 0.0),
    ([5, 3], [False, True], 0.0), ([1500, 700, 64, 3000], [True, False, True, True], -6.0),
    ([200 + 17 * i for i in range(16)], [i % 3 != 0 for i in range(16)], 0.0), ([4096, 2500], [True, True], -3.0),
    ([2500, 1700], [False, True], -6.0), ([64 + 3 * i for i in range(40)], [i % 2 == 0 for i in range(40)], 0.0),
], ids=["64", "130_noinit", "2048", "ragged4", "tiny", "ragged_slow", "sixteen", "two_long", "two_slow", "forty"])
@torch.inference_mode()
def test_prefill(impl, seq, init, dt_shift):
    require_gfx950()
    c = make_case(seq, init=init, dt_shift=dt_shift)
    ref, conv_state, state = reference(c)
    out, got_conv, got_state = run(c, impl)
    assert not torch.isnan(out.float()).any()
    assert rel(out, ref) < OUT_TOL
    assert rel(got_state, state) < STATE_TOL
    assert torch.equal(got_conv, conv_state)
    again, _, _ = run(c, impl)
    assert torch.equal(out, again)


@pytest.mark.parametrize("impl", IMPLS)
@torch.inference_mode()
def test_state_continuation(impl):
    """Two prefill calls over the halves of a sequence (the second with the state and conv history the first left) equal one call."""
    require_gfx950()
    first, second = 640, 384
    c = make_case([first + second], init=[False])
    whole_out, whole_conv, whole_state = run(c, impl)
    outs, conv_state, state = [], c["conv_state"].clone(), c["state_cache"].clone()
    for lo, hi, init in ((0, first, False), (first, first + second, True)):
        sub = dict(c)
        sub["mixed"], sub["raw_g"] = c["mixed"][lo:hi], c["raw_g"][lo:hi]
        sub["beta_logit"], sub["gate"] = c["beta_logit"][lo:hi], c["gate"][lo:hi]
        cu = torch.tensor([0, hi - lo], dtype=torch.int32, device="cuda")
        out = torch.zeros(hi - lo, R.NH, R.D, dtype=torch.bfloat16, device="cuda")
        host.prefill(sub["mixed"], sub["raw_g"], sub["beta_logit"], sub["gate"], c["A_log"], c["dt_bias"], c["conv_weight"], conv_state,
                     state, c["state_indices"], torch.tensor([init], device="cuda"), cu, chunk_indices(cu), offsets(cu), c["norm_weight"],
                     c["norm_eps"], out, c["lower_bound"], chunk_counts=counts(cu), impl=impl)
        outs.append(out)
    assert torch.equal(torch.cat(outs), whole_out)
    assert torch.equal(state, whole_state) and torch.equal(conv_state, whole_conv)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("seq,offs,rows,dt_shift", [
    ([640], [320], None, 0.0), ([2048, 300], [1024, 256], [3, 1], -6.0), ([640, 700], [640, 0], [1, 0], 0.0),
], ids=["single", "ragged_rows", "at_end_and_off"])
@torch.inference_mode()
def test_checkpoint(impl, seq, offs, rows, dt_shift):
    """The state after ``checkpoint_offsets[n]`` tokens lands in the checkpoint rows; the final states are unchanged."""
    require_gfx950()
    c = make_case(seq, dt_shift=dt_shift)
    ck_off = torch.tensor(offs, dtype=torch.int32, device="cuda")
    ck_rows = torch.tensor(rows if rows is not None else list(range(len(seq))), dtype=torch.int32, device="cuda")
    ref_ck = torch.zeros(len(seq) + 2, R.NH, R.D, R.D, device="cuda")
    _, _, want_state = reference(c, checkpoint_state=ref_ck, checkpoint_offsets=ck_off, checkpoint_rows=ck_rows)
    ck = torch.zeros_like(ref_ck)
    _, _, state = run(c, impl, checkpoint_state=ck, checkpoint_offsets=ck_off, checkpoint_rows=None if rows is None else ck_rows)
    assert rel(state, want_state) < STATE_TOL
    assert rel(ck, ref_ck) < STATE_TOL
    touched = {r for r, o in zip(ck_rows.tolist(), offs) if o > 0}
    for r in range(ck.shape[0]):
        if r not in touched:
            assert not ck[r].any()


@torch.inference_mode()
def test_one_launch_hand_over():
    """Back-to-back calls of the one-launch path leave every hand-over flag reset and give the same bits; a walk workgroup cap far
    below the number of (sequence, head, slice group) units only serializes them."""
    require_gfx950()
    c = make_case([200 + 17 * i for i in range(12)])
    cu = c["cu_seqlens"]
    outs = []
    for cap in (None, None, 32, 32, 128):
        out = torch.zeros(int(cu[-1]), R.NH, R.D, dtype=torch.bfloat16, device="cuda")
        host.prefill_mega(c["mixed"], c["raw_g"], c["beta_logit"], c["gate"], c["A_log"], c["dt_bias"], c["conv_weight"],
                          c["conv_state"].clone(), c["state_cache"].clone(), c["state_indices"], c["has_initial_state"], cu,
                          chunk_indices(cu), offsets(cu), c["norm_weight"], c["norm_eps"], out, c["lower_bound"], walk_wgs=cap)
        torch.cuda.synchronize()
        assert not host._flags(out.device, chunk_indices(cu).shape[0]).any()
        outs.append(out)
    assert all(torch.equal(outs[0], o) for o in outs[1:])


def test_dispatch():
    assert host.uses_mega([32]) and host.uses_mega([60]) and host.uses_mega([1]) and host.uses_mega([16, 16]) and host.uses_mega([5, 1, 2, 1])
    assert host.uses_mega([19, 19, 10]) and host.uses_mega([2] * 8)
    assert not host.uses_mega([64]) and not host.uses_mega([11] * 4) and not host.uses_mega([4] * 8) and not host.uses_mega([30, 31])
    assert not host.uses_mega(None) and not host.uses_mega([2] * 9)


@pytest.mark.parametrize("impl", IMPLS)
@torch.inference_mode()
def test_conv_state_slots_past_2gb(impl):
    """vLLM's hybrid cache lays every conv-state slot inside a ~21 MB page (slot stride ~10M bf16, dim-first view of a
    [slot, pos, dim] block). Slots whose byte offset passes 2 GB (and 4 GB, where a 32-bit offset wraps) must read and
    roll their own row: the served model rolled the wrong (or no) row for high slots before the 64-bit slot base."""
    require_gfx950()
    seq = [150, 70, 200, 64]
    c = make_case(seq, init=[True, False, True, True], slots=4)
    stride = 1 << 24  # elements: slot 65+ is past 2 GB, slot 128+ past 4 GB
    slots = [3, 66, 129, 140]
    pool = torch.zeros(141 * stride, dtype=torch.bfloat16, device="cuda")
    cs = pool.as_strided((141, 3 * R.LP, 3), (stride, 1, 3 * R.LP))  # dim-first view of [slot, pos, dim] rows
    small = c["conv_state"]
    for k, s in enumerate(slots):
        cs[s].copy_(small[int(c["state_indices"][k])])
    sentinel = (torch.randn(3 * R.LP, 3, device="cuda") * 0.5).bfloat16()
    for s in (2, 65, 128):
        cs[s].copy_(sentinel)
    st = torch.zeros(141, R.NH, R.D, R.D, dtype=torch.float32, device="cuda")
    for k, s in enumerate(slots):
        st[s].copy_(c["state_cache"][int(c["state_indices"][k])])
    sidx = torch.tensor(slots, dtype=torch.int32, device="cuda")
    cu = c["cu_seqlens"]
    ref_cs, ref_st = cs.clone(), st.clone()
    o_ref = R.kda_prefill_reference(c["mixed"], c["raw_g"], c["beta_logit"], c["gate"], c["A_log"], c["dt_bias"], c["lower_bound"],
                                    c["conv_weight"], ref_cs, ref_st, sidx, c["has_initial_state"], cu, c["norm_weight"], c["norm_eps"])
    out = torch.zeros(int(cu[-1]), R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    host.prefill(c["mixed"], c["raw_g"], c["beta_logit"], c["gate"], c["A_log"], c["dt_bias"], c["conv_weight"], cs, st, sidx,
                 c["has_initial_state"], cu, chunk_indices(cu), offsets(cu), c["norm_weight"], c["norm_eps"], out, c["lower_bound"],
                 chunk_counts=counts(cu), impl=impl)
    torch.cuda.synchronize()
    assert rel(out, o_ref) < OUT_TOL
    for s in slots:
        assert torch.equal(cs[s], ref_cs[s]), f"conv state of slot {s} not rolled like the reference"
        assert rel(st[s], ref_st[s]) < STATE_TOL
    for s in (2, 65, 128):
        assert torch.equal(cs[s], sentinel), f"slot {s} (no sequence) was written"
    del pool
