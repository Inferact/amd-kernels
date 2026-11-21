# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The state walk and the output pass against reference.chunk_reference / gated_norm_reference on workspaces built by reference.pack_record."""

import pytest
import torch

from amd_kernels.attention.kda_prefill_h12_d128 import host
from amd_kernels.attention.kda_prefill_h12_d128 import prologue as P
from amd_kernels.attention.kda_prefill_h12_d128 import reference as R

from .inputs import make_case, require_gfx950

OUT_TOL = 6e-3
STATE_TOL = 4e-3


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def workspace(c):
    cu = c["cu_seqlens"]
    conv, _ = R.conv_reference(c["mixed"], c["conv_weight"], c["conv_state"], c["state_indices"], c["has_initial_state"], cu)
    pro = R.prologue_reference(conv[:, :R.LP], conv[:, R.LP:2 * R.LP], conv[:, 2 * R.LP:], c["raw_g"], c["beta_logit"],
                               c["A_log"], c["dt_bias"], c["lower_bound"], R.SCALE, cu)
    return R.pack_record(pro)


def tables(cu, c):
    rows = [(n, k) for n, k, _, _ in R.chunk_table(cu)[0]]
    ci = torch.tensor(rows, dtype=torch.int32, device="cuda")
    co = torch.tensor(R.chunk_table(cu)[1], dtype=torch.int64, device="cuda")
    return P.chunk_table(cu, ci, c["state_indices"], c["has_initial_state"]), co


def run(c, rec, **kw):
    cu = c["cu_seqlens"]
    table, co = tables(cu, c)
    state = c["state_cache"].clone()
    sb = host.walk_states(rec, cu, co, state, c["state_indices"], c["has_initial_state"], **kw)
    out = torch.zeros(int(cu[-1]), R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    host.chunk_outputs(rec, sb, table, c["gate"], c["norm_weight"], c["norm_eps"], out)
    return out, state


def reference(c, rec, **kw):
    cu = c["cu_seqlens"]
    state = c["state_cache"].clone()
    o = R.chunk_reference(R.unpack_record(rec), cu, state, c["state_indices"], c["has_initial_state"], **kw)
    return R.gated_norm_reference(o.bfloat16(), c["gate"], c["norm_weight"], c["norm_eps"]), state


@pytest.mark.parametrize("seq,init,dt_shift", [
    ([64], [True], 0.0), ([130], [False], 0.0), ([2048], [True], 0.0), ([300, 64, 77, 1], [True, False, True, True], 0.0),
    ([5, 3], [False, True], 0.0), ([1500, 700, 64, 3000], [True, False, True, True], -6.0), ([4096], [True], -6.0),
    ([200 + 17 * i for i in range(16)], [i % 3 != 0 for i in range(16)], 0.0),
], ids=["64", "130_noinit", "2048", "ragged4", "tiny", "ragged_slow", "4096_slow", "sixteen"])
@torch.inference_mode()
def test_walk_out(seq, init, dt_shift):
    require_gfx950()
    c = make_case(seq, init=init, dt_shift=dt_shift)
    rec = workspace(c)
    want, want_state = reference(c, rec)
    out, state = run(c, rec)
    assert rel(out, want) < OUT_TOL
    assert rel(state, want_state) < STATE_TOL
    again, state2 = run(c, rec)
    assert torch.equal(out, again) and torch.equal(state, state2)


@torch.inference_mode()
def test_state_continuation():
    """Two walks over the halves of a sequence (the second from the state the first left) equal one walk over the whole."""
    require_gfx950()
    first, second = 640, 384
    c = make_case([first + second], init=[False])
    rec = workspace(c)
    out_whole, state_whole = run(c, rec)
    nc1 = first // 64
    rec1, rec2 = rec[:, :nc1].contiguous(), rec[:, nc1:].contiguous()
    gate = c["gate"]
    st = c["state_cache"].clone()
    outs = []
    for part, (lo, hi), has in ((rec1, (0, first), c["has_initial_state"]), (rec2, (first, first + second), torch.ones_like(c["has_initial_state"]))):
        cu_p = torch.tensor([0, hi - lo], dtype=torch.int32, device="cuda")
        ci = torch.tensor([[0, k] for k in range(part.shape[1])], dtype=torch.int32, device="cuda")
        co = torch.tensor([0, part.shape[1]], dtype=torch.int64, device="cuda")
        table = P.chunk_table(cu_p, ci, c["state_indices"], has)
        sb = host.walk_states(part, cu_p, co, st, c["state_indices"], has)
        o = torch.zeros(hi - lo, R.NH, R.D, dtype=torch.bfloat16, device="cuda")
        host.chunk_outputs(part, sb, table, gate[lo:hi], c["norm_weight"], c["norm_eps"], o)
        outs.append(o)
    assert torch.equal(torch.cat(outs), out_whole)
    assert torch.equal(st, state_whole)


@pytest.mark.parametrize("seq,offs,rows,dt_shift", [
    ([640], [320], None, 0.0), ([2048, 300], [1024, 256], [3, 1], -6.0), ([640, 700], [640, 0], [1, 0], 0.0), ([1500], [1472], None, -6.0),
], ids=["single", "ragged_rows", "at_end_and_off", "late"])
@torch.inference_mode()
def test_checkpoint(seq, offs, rows, dt_shift):
    """The state after ``checkpoint_offsets[n]`` tokens lands in the checkpoint rows; the final states are unchanged."""
    require_gfx950()
    c = make_case(seq, dt_shift=dt_shift)
    rec = workspace(c)
    ck_off = torch.tensor(offs, dtype=torch.int32, device="cuda")
    ck_rows = torch.tensor(rows if rows is not None else list(range(len(seq))), dtype=torch.int32, device="cuda")
    ref_ck = torch.zeros(len(seq) + 2, R.NH, R.D, R.D, device="cuda")
    _, want_state = reference(c, rec, checkpoint_state=ref_ck, checkpoint_offsets=ck_off, checkpoint_rows=ck_rows)
    ck = torch.zeros_like(ref_ck)
    _, state = run(c, rec, checkpoint_state=ck, checkpoint_offsets=ck_off, checkpoint_rows=None if rows is None else ck_rows)
    assert rel(state, want_state) < STATE_TOL
    assert rel(ck, ref_ck) < STATE_TOL
    touched = {r for r, o in zip(ck_rows.tolist(), offs) if o > 0}
    for r in range(ck.shape[0]):
        if r not in touched:
            assert not ck[r].any()
