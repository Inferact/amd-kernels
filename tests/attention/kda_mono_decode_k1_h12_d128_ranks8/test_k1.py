# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""K1 against reference.py at the served widths (12 and 6 rows of 6-token requests, and the other (rows, qlen) pairs
up to 16 rows), the block-write layer, a missing delta, both conv-state layouts, graph padding requests, run-to-run
determinism, and a CUDA graph of two launches replayed over poisoned outputs."""

import pytest
import torch
from amd_kernels.attention.kda_mono_decode_k1_h12_d128_ranks8 import Workspace, kda_mono_decode_k1

from .workload import clone_state, cos, make, reference, rel_l2, require_gfx950

SHAPES = [(12, 6), (6, 6), (8, 4), (16, 8), (12, 3), (4, 4), (3, 3), (16, 4), (2, 2)]


def launch(ws, layer, w, L, nb, write_idx=-1, delta=True):
    s = clone_state(w)
    core = kda_mono_decode_k1(
        ws, layer, s["prefix"], s["delta"] if delta else None, s["blocks"], s["ares_nw"], s["ares_qk"], s["in_nw"],
        s["w_in"], s["w_fb"], s["conv_w"], s["conv_state"], s["a_log"], s["dt_bias"], s["on_w"], s["rstate"],
        s["st_idx"], s["num_acc"], qlen=L, nblocks=nb, write_idx=write_idx,
    )  # fmt: skip
    return s, core


def check(got, core, ref, ref_core):
    assert torch.equal(got["prefix"], ref["prefix"]) and torch.equal(got["blocks"], ref["blocks"])
    assert rel_l2(got["conv_state"], ref["conv_state"]) < 1e-3
    assert rel_l2(got["rstate"], ref["rstate"]) < 1e-3
    assert rel_l2(core, ref_core) < 3e-3 and cos(core, ref_core) > 0.9999


@pytest.mark.parametrize("S,L", SHAPES)
@pytest.mark.parametrize("nb", [1, 4, 8])
@pytest.mark.parametrize("dim_first", [True, False])
def test_matches_reference(S, L, nb, dim_first):
    require_gfx950()
    w = make(S, L, nb, nacc=2, dim_first=dim_first)
    ws = Workspace("cuda")
    ws.begin_step()
    got, core = launch(ws, 0, w, L, nb)
    torch.cuda.synchronize()
    assert not ws.failed()
    check(got, core, *reference(w, L, nb, -1))


@pytest.mark.parametrize("S,L", [(12, 6), (6, 6)])
def test_block_write_and_no_delta(S, L):
    require_gfx950()
    nb = 4
    w = make(S, L, nb)
    ws = Workspace("cuda")
    ws.begin_step()
    got, core = launch(ws, 0, w, L, nb, write_idx=nb)
    check(got, core, *reference(w, L, nb, nb))
    ws.begin_step()
    s = clone_state(w)
    s["delta"] = None
    from amd_kernels.attention.kda_mono_decode_k1_h12_d128_ranks8 import reference as R

    ref = clone_state(w)
    ref_core = R.kda_decode_k1(
        ref["prefix"], None, ref["blocks"], ref["ares_nw"], ref["ares_qk"], ref["in_nw"], ref["w_in"], ref["w_fb"],
        ref["conv_w"], ref["conv_state"], ref["a_log"], ref["dt_bias"], ref["on_w"], ref["rstate"], ref["st_idx"],
        ref["num_acc"], qlen=L, nblocks=nb,
    )  # fmt: skip
    got, core = launch(ws, 1, w, L, nb, delta=False)
    check(got, core, ref, ref_core)


def test_graph_padding_requests_are_inert():
    require_gfx950()
    S, L, nb = 12, 6, 4
    w = make(S, L, nb, null_requests=1)
    ws = Workspace("cuda")
    ws.begin_step()
    got, core = launch(ws, 0, w, L, nb)
    ref, ref_core = reference(w, L, nb, -1)
    check(got, core, ref, ref_core)
    assert not core[L:].any()
    assert torch.equal(got["rstate"][0], w["rstate"][0])


def test_deterministic():
    require_gfx950()
    S, L, nb = 12, 6, 4
    w = make(S, L, nb, nacc=3)
    ws = Workspace("cuda")
    outs = []
    for i in range(3):
        ws.begin_step()
        outs.append(launch(ws, 0, w, L, nb))
    for got, core in outs[1:]:
        assert torch.equal(core, outs[0][1]) and torch.equal(got["rstate"], outs[0][0]["rstate"])


def test_graph_replay_over_poison():
    require_gfx950()
    S, L, nb = 12, 6, 4
    w = make(S, L, nb)
    ws = Workspace("cuda")
    s0, s1 = clone_state(w), clone_state(w)
    ref0, rc0 = reference(w, L, nb, -1)

    def step(s):
        ws.begin_step()
        c0 = kda_mono_decode_k1(
            ws, 0, s["prefix"], s["delta"], s["blocks"], s["ares_nw"], s["ares_qk"], s["in_nw"], s["w_in"],
            s["w_fb"], s["conv_w"], s["conv_state"], s["a_log"], s["dt_bias"], s["on_w"], s["rstate"], s["st_idx"],
            s["num_acc"], qlen=L, nblocks=nb,
        )  # fmt: skip
        return c0

    step(clone_state(w))  # eager: compiles
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        core = step(s0)
    for _ in range(5):
        for k in ("prefix", "blocks", "conv_state", "rstate"):
            s0[k].copy_(w[k])
        core.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        assert not ws.failed()
        check(s0, core, ref0, rc0)
