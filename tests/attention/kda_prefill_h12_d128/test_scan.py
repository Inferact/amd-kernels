# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The chunk scan against reference.chunk_reference on workspaces built by reference.pack_record."""

import pytest
import torch
from amd_kernels.attention.kda_prefill_h12_d128 import host
from amd_kernels.attention.kda_prefill_h12_d128 import reference as R

from .inputs import make_case, require_gfx950

OUT_TOL = 6e-3
STATE_TOL = 4e-3


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def workspace(c):  # noqa: D103
    cu = c["cu_seqlens"]
    conv, _ = R.conv_reference(c["mixed"], c["conv_weight"], c["conv_state"], c["state_indices"], c["has_initial_state"], cu)
    pro = R.prologue_reference(conv[:, :R.LP], conv[:, R.LP:2 * R.LP], conv[:, 2 * R.LP:], c["raw_g"], c["beta_logit"],
                               c["A_log"], c["dt_bias"], c["lower_bound"], R.SCALE, cu)
    return R.pack_record(pro), pro


def offsets(cu):
    return torch.tensor(R.chunk_table(cu)[1], dtype=torch.int64, device=cu.device)


@pytest.mark.parametrize("seq,init,groups,dt_shift", [
    ([64], [True], 1, 0.0), ([130], [False], 1, 0.0), ([2048], [True], 1, 0.0), ([300, 64, 77, 1], [True, False, True, True], 1, 0.0),
    ([5, 3], [False, True], 1, 0.0), ([2048], [True], 4, 0.0), ([2048], [True], 4, -6.0), ([2048], [False], 8, -6.0),
    ([1500, 700, 64, 3000], [True, False, True, True], 6, -6.0), ([4096], [True], 10, -6.0),
], ids=["64", "130_noinit", "2048", "ragged4", "tiny", "2048_g4", "2048_g4_slow", "2048_noinit_g8_slow", "ragged_g6_slow", "4096_g10_slow"])
@torch.inference_mode()
def test_scan(seq, init, groups, dt_shift):
    require_gfx950()
    c = make_case(seq, init=init, dt_shift=dt_shift)
    rec, pro = workspace(c)
    cu = c["cu_seqlens"]
    ref_state = c["state_cache"].clone()
    o_ref = R.chunk_reference(R.unpack_record(rec), cu, ref_state, c["state_indices"], c["has_initial_state"])
    state = c["state_cache"].clone()
    out = torch.zeros(sum(seq), R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    host.chunk_scan(rec, cu, offsets(cu), state, c["state_indices"], c["has_initial_state"], out, groups=groups)
    assert rel(out, o_ref) < OUT_TOL
    assert rel(state, ref_state) < STATE_TOL
    again = c["state_cache"].clone()
    out2 = torch.zeros_like(out)
    host.chunk_scan(rec, cu, offsets(cu), again, c["state_indices"], c["has_initial_state"], out2, groups=groups)
    assert torch.equal(out, out2) and torch.equal(state, again)


@torch.inference_mode()
def test_state_continuation():
    """Two calls over the halves of a sequence (second with the state the first left) equal one call over the whole."""
    require_gfx950()
    first, second = 640, 384
    c = make_case([first + second], init=[False])
    rec, pro = workspace(c)
    cu = c["cu_seqlens"]
    whole_state = c["state_cache"].clone()
    out_whole = torch.zeros(first + second, R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    host.chunk_scan(rec, cu, offsets(cu), whole_state, c["state_indices"], c["has_initial_state"], out_whole)
    ws = R.unpack_record(rec)
    nc1 = first // 64
    rec1, rec2 = rec[:, :nc1].contiguous(), rec[:, nc1:].contiguous()
    cu1 = torch.tensor([0, first], dtype=torch.int32, device="cuda")
    cu2 = torch.tensor([0, second], dtype=torch.int32, device="cuda")
    st = c["state_cache"].clone()
    o1 = torch.zeros(first, R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    o2 = torch.zeros(second, R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    host.chunk_scan(rec1, cu1, offsets(cu1), st, c["state_indices"], c["has_initial_state"], o1)
    host.chunk_scan(rec2, cu2, offsets(cu2), st, c["state_indices"], torch.ones_like(c["has_initial_state"]), o2)
    assert torch.equal(torch.cat([o1, o2]), out_whole)
    assert torch.equal(st, whole_state)


@pytest.mark.parametrize("seq,offs,rows,groups,dt_shift", [
    ([640], [320], None, 1, 0.0), ([2048, 300], [1024, 256], [3, 1], 4, -6.0), ([640, 700], [640, 0], [1, 0], 1, 0.0),
    ([1500], [1472], None, 5, -6.0),
], ids=["single", "ragged_rows_g4", "at_end_and_off", "late_g5"])
@torch.inference_mode()
def test_checkpoint(seq, offs, rows, groups, dt_shift):
    """The state after ``checkpoint_offsets[n]`` tokens lands in the checkpoint rows; the final states are unchanged."""
    require_gfx950()
    c = make_case(seq, dt_shift=dt_shift)
    rec, _ = workspace(c)
    cu = c["cu_seqlens"]
    ref_state, ref_ck = c["state_cache"].clone(), torch.zeros(len(seq) + 2, R.NH, R.D, R.D, device="cuda")
    ck_off = torch.tensor(offs, dtype=torch.int32, device="cuda")
    ck_rows = torch.tensor(rows if rows is not None else list(range(len(seq))), dtype=torch.int32, device="cuda")
    R.chunk_reference(R.unpack_record(rec), cu, ref_state, c["state_indices"], c["has_initial_state"], ref_ck, ck_off, ck_rows)
    state, ck = c["state_cache"].clone(), torch.zeros_like(ref_ck)
    out = torch.zeros(sum(seq), R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    host.chunk_scan(rec, cu, offsets(cu), state, c["state_indices"], c["has_initial_state"], out, groups=groups, checkpoint_state=ck,
              checkpoint_offsets=ck_off, checkpoint_rows=None if rows is None else ck_rows)
    assert rel(state, ref_state) < STATE_TOL
    assert rel(ck, ref_ck) < STATE_TOL
    touched = {r for r, o in zip(ck_rows.tolist(), offs) if o > 0}
    for r in range(ck.shape[0]):
        if r not in touched:
            assert not ck[r].any()
