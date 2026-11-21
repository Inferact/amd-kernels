# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entries and the tile tiers of the DCP8 query path."""

import torch

from amd_kernels import warm_up
from .comm import MAX_ROWS

NAME_LATENT = "attention/dcp8_q_bproj_wuk_gather_h96_kv512/latent"
NAME_Q = "attention/dcp8_q_bproj_wuk_gather_h96_kv512/q"

# (max rows, q_b tile (mt, nt, nw), W_UK tile (mt, ns)) of the latent entry; (max rows, q_b tile (mt, nt), W_UK
# workgroups per head, consumer delay) of the q entry: sweeps of benchmarks/attention/dcp8_q_bproj_wuk_gather_h96_kv512
TIERS_LATENT = (
    (16, (1, 1, 4), (1, 4)),
    (48, (1, 2, 4), (1, 4)),
    (96, (2, 2, 4), (2, 4)),
    (MAX_ROWS, (2, 4, 4), (3, 4)),
)
# (max rows, q_b column tiles per producer, producers per column block, waves per workgroup, push stagger) of the q
# entry: sweeps of benchmarks/attention/dcp8_q_bproj_wuk_gather_h96_kv512 (one tier compiles per row bound)
TIERS_Q = (
    (16, 1, 1, 8, 0),
    (32, 2, 2, 8, 0),
    (48, 2, 3, 4, 0),
    (64, 2, 3, 4, 0),
    (80, 2, 3, 4, 0),
    (96, 4, 4, 8, 0),
    (128, 4, 3, 8, 3),
    (160, 4, 3, 8, 3),
    (192, 4, 3, 8, 3),
    (MAX_ROWS, 4, 4, 8, 0),
)


def config_latent(rows):
    for top, qb, wuk in TIERS_LATENT:
        if rows <= top:
            return qb, wuk
    raise ValueError(rows)


def config_q(rows):
    for top, nt1, nsplit, nw, stag in TIERS_Q:
        if rows <= top:
            return nt1, nsplit, nw, top, stag
    raise ValueError(rows)


def _check(q_a, w_qb, w_uk, heads):
    m = q_a.shape[0]
    assert 0 < m <= MAX_ROWS and q_a.shape[1] == 1536 and q_a.dtype == torch.bfloat16
    assert q_a.stride(1) == 1 and q_a.stride(0) % 8 == 0
    assert w_qb.dtype == w_uk.dtype == torch.bfloat16 and w_qb.is_contiguous() and w_uk.is_contiguous()
    assert w_uk.numel() == heads * 128 * 512
    return m


def _mode(comm, q_a, w_qb, w_uk):
    return ()


def _sizes(mode):
    return tuple(top for top, *_ in TIERS_Q)


def _inputs(mode, n, comm, q_a, w_qb, w_uk):
    return (comm, torch.zeros((n, 1536), dtype=q_a.dtype, device=q_a.device), w_qb, w_uk), {}


@warm_up.entry(NAME_LATENT, mode=_mode, sizes=_sizes, inputs=_inputs)
def q_b_wuk_allgather_latent(comm, q_a, w_qb, w_uk):
    """Latent query [M, 96, 576] of all 96 heads: this rank's q_b + W_UK heads, pushed to every rank."""
    assert comm.mode == "latent"
    m = _check(q_a, w_qb, w_uk, 12)
    return comm._latent(q_a, w_qb, w_uk, config_latent(m))


def _mode_q(comm, q_a, w_qb, w_uk96, out=None, out_q8=None):
    return ("bf16",) if out_q8 is None else ("q8",) if out is None else ("both",)


def _inputs_q(mode, n, comm, q_a, w_qb, w_uk96, out=None, out_q8=None):
    dev = q_a.device
    kw = {}
    if mode[0] != "q8":
        kw["out"] = torch.zeros((n, 96, 576), dtype=torch.bfloat16, device=dev)
    if mode[0] != "bf16":
        kw["out_q8"] = torch.zeros((n, 96, 1168), dtype=torch.uint8, device=dev)
    return (comm, torch.zeros((n, 1536), dtype=q_a.dtype, device=dev), w_qb, w_uk96), kw


@warm_up.entry(NAME_Q, mode=_mode_q, sizes=_sizes, inputs=_inputs_q)
def q_b_allgather_wuk_all_heads(comm, q_a, w_qb, w_uk96, out=None, out_q8=None):
    """The query of all 96 heads on every rank: this rank's q_b tiles are pushed to all eight ranks, which apply the
    W_UK of all 96 heads (replicated weight) and write ``out`` bf16 [M, 96, 576] = ``[bf16(q_nope W_UK) | q_pe]``
    and/or ``out_q8`` uint8 [M, 96, 1168], the hi/lo fp8 record of ``reference.quantize_q`` (the sparse-MLA decode
    kernel's pre-quantized q). Neither given: ``out`` is allocated. Returns ``out``, ``out_q8`` or ``(out, out_q8)``."""
    assert comm.mode == "q"
    m = _check(q_a, w_qb, w_uk96, 96)
    if out is None and out_q8 is None:
        out = torch.empty((m, 96, 576), dtype=torch.bfloat16, device=q_a.device)
    assert out is None or (out.shape == (m, 96, 576) and out.dtype == torch.bfloat16 and out.is_contiguous())
    assert out_q8 is None or (out_q8.shape == (m, 96, 1168) and out_q8.dtype == torch.uint8 and out_q8.is_contiguous())
    fp8out = 0 if out_q8 is None else 1 if out is None else 2
    comm._push(q_a, w_qb, w_uk96, out, out_q8, config_q(m), fp8out)
    return out if out_q8 is None else out_q8 if out is None else (out, out_q8)
