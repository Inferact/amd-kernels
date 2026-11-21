# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Served-shape calls of the KDA entries and of the shape packages they route to (tests and benchmark): each builder
returns fresh, seeded inputs, so two builds are equal bit for bit."""

import importlib

import torch

from amd_kernels.attention import kda_common as E
from amd_kernels.attention.kda_prefill_h12_d128 import reference as PR

from tests.attention.kda_decode_fb_gate_spec_h12_d128.inputs import make_inputs
from tests.attention.kda_mono_decode_k1_h12_d128_ranks8.workload import make as make_mono
from tests.attention.kda_prefill_h12_d128.inputs import make_case

DECODE_PKG = "amd_kernels.attention.kda_decode_fb_gate_spec_h12_d128"
PREFILL_PKG = "amd_kernels.attention.kda_prefill_h12_d128"
MONO_PKG = "amd_kernels.attention.kda_mono_decode_k1_h12_d128_ranks8"


class Call:
    """One call: ``run(entry)`` launches it through the entry (True) or the shape package (False); ``outputs``
    are every tensor the call writes."""

    def __init__(self, run, outputs):
        self.run, self.outputs = run, outputs


def decode(rows, tokens, uniform, seed=0):
    a = make_inputs(rows, tokens=tokens, seed=seed)
    pkg = importlib.import_module(DECODE_PKG)

    def run(entry):
        fn = E.kda_decode_fb_gate_spec if entry else pkg.kda_decode_fb_gate_spec
        fn(**a, uniform=uniform)

    return Call(run, lambda: [a["out"], a["state"], a["conv_state"]]), a


def prefill(seq, impl=None, checkpoint=None, seed=0):
    """``checkpoint``: (offsets, rows) of the checkpoint export, or None."""
    c = make_case(seq, seed=seed)
    cu = c["cu_seqlens"]
    rows, offs = PR.chunk_table(cu)
    ci = torch.tensor([[n, k] for n, k, _, _ in rows], dtype=torch.int32, device="cuda")
    co = torch.tensor(offs, dtype=torch.int64, device="cuda")
    out = torch.zeros(int(cu[-1]), PR.NH, PR.D, dtype=torch.bfloat16, device="cuda")
    args = (c["mixed"], c["raw_g"], c["beta_logit"], c["gate"], c["A_log"], c["dt_bias"], c["conv_weight"],
            c["conv_state"], c["state_cache"], c["state_indices"], c["has_initial_state"], cu, ci, co, c["norm_weight"],
            c["norm_eps"], out, c["lower_bound"])
    kw = dict(chunk_counts=[b - a for a, b in zip(offs[:-1], offs[1:])], impl=impl)
    outputs = [out, c["conv_state"], c["state_cache"]]
    if checkpoint is not None:
        ck = torch.zeros(len(seq) + 2, PR.NH, PR.D, PR.D, device="cuda")
        kw.update(checkpoint_state=ck, checkpoint_offsets=torch.tensor(checkpoint[0], dtype=torch.int32, device="cuda"),
                  checkpoint_rows=torch.tensor(checkpoint[1], dtype=torch.int32, device="cuda"))
        outputs.append(ck)
    pkg = importlib.import_module(PREFILL_PKG)
    table_args = (cu, ci, c["state_indices"], c["has_initial_state"])

    def run(entry):
        if entry:
            table = E.kda_prefill_chunk_table(*table_args, state_cache=c["state_cache"])
            E.kda_prefill(*args, table=table, **kw)
        else:
            pkg.prefill(*args, table=pkg.chunk_table(*table_args), **kw)

    return Call(run, lambda: outputs), (args, kw)


def mono(S, L, nb=4, write_idx=-1, delta=True, seed=0):
    w = make_mono(S, L, nb, nacc=2, seed=seed)
    if not delta:
        w["delta"] = None
    pkg = importlib.import_module(MONO_PKG)
    ws = {True: E.kda_mono_workspace("cuda", hidden=7168, heads=12, head_dim=128), False: pkg.Workspace("cuda")}
    core = []
    order = ("prefix", "delta", "blocks", "ares_nw", "ares_qk", "in_nw", "w_in", "w_fb", "conv_w", "conv_state",
             "a_log", "dt_bias", "on_w", "rstate", "st_idx", "num_acc")
    args = [w[k] for k in order]
    kw = dict(qlen=L, nblocks=nb, write_idx=write_idx)

    def run(entry):
        ws[entry].begin_step()
        fn = E.kda_mono_decode if entry else pkg.kda_mono_decode_k1
        core[:] = [fn(ws[entry], 0, *args, **kw)]

    return Call(run, lambda: [core[0], w["prefix"], w["blocks"], w["conv_state"], w["rstate"]]), (ws, args, kw)
