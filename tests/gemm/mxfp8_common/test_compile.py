# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4).

A serving engine warms each configuration, then sends any M the configuration serves.
A configuration launches a finite set of variants chosen by M: the activation
quantizer (the FlyDSL kernel's rows per workgroup) and the four-wave tail tiles; ``variant``
names that choice from the host-side plan. Per case: warm one M per variant over the served range, then probe other Ms; a
probe that compiles means M itself (or something unbounded derived from it) reaches a
compile key, so a server would compile on the forward path.
"""

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common import mxfp8_quant
from amd_kernels.gemm.mxfp8_common.gemm import _fourwave_plan, mxfp8_linear

from .test_gemm import CASES, make_inputs, require_gfx950

# the M range each case's configuration serves: its row of the M3.1 table or window
RANGES = {
    "aiter": range(257, 385),
    "decode_fq": range(1, 5),
    "decode_fq48": range(17, 33),
    "decode": range(33, 65),
    "decode48": range(193, 257),
    "decode_kw4": range(289, 385),
    "decode_kw4_rt7": range(385, 417),
    "staged": range(769, 1025),
    "fourwave": range(769, 1281),
    "fourwave_nt": range(1281, 16385),
}
PROBES = 6


def quantizer(rows, shuffled):
    """The activation quantizer ``mxfp8_linear`` launches for ``rows`` rows: the FlyDSL
    kernel compiled per rows-per-workgroup (``mxfp8_quant.quantize``)."""
    prefill = rows >= mxfp8_quant.PREFILL_ROWS and not shuffled
    return mxfp8_quant.PREFILL_RPW if prefill else mxfp8_quant.DEFAULT_RPW


def variant(name, m, n, k, config):
    """The part of a launch that picks a compiled variant."""
    if config.kernel == "decode_fq":
        return None  # the kernel quantizes the bf16 activation itself
    if config.kernel == "decode":
        return quantizer(-(-m // config.tile_m) * config.tile_m, shuffled=True)
    quant = quantizer(m, shuffled=False)
    if config.kernel in ("4wave", "4wave_nt"):
        return quant, tuple(tiles for tiles, _, _ in _fourwave_plan(m, n, config))
    return quant


def warm_and_probe(served, key):
    warm = {}
    for rows in served:
        warm.setdefault(key(rows), rows)
    rest = [rows for rows in served if rows not in warm.values()]
    return warm, rest[:: max(1, len(rest) // PROBES)][:PROBES]


def count_compiles(monkeypatch):
    compiled = []
    real_compile = flyc.compile

    def counting_compile(fn, *args, **kwargs):
        compiled.append(getattr(fn, "__name__", type(fn).__name__))
        return real_compile(fn, *args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting_compile)
    return compiled


@pytest.mark.parametrize("name,m,n,k,config", CASES, ids=[c[0] for c in CASES])
@torch.inference_mode()
def test_no_compile_after_warmup(name, m, n, k, config, monkeypatch):
    require_gfx950()
    served = RANGES[name]
    warm, probe = warm_and_probe(served, lambda rows: variant(name, rows, n, k, config))
    x, _, _, (w, w_scale) = make_inputs(served[-1], n, k)
    for rows in warm.values():
        mxfp8_linear(x[:rows], w, w_scale, config=config)
    torch.cuda.synchronize()

    compiled = count_compiles(monkeypatch)
    for rows in probe:
        mxfp8_linear(x[:rows], w, w_scale, config=config)
        assert not compiled, (
            f"{name}: M={rows} (variant {variant(name, rows, n, k, config)}) compiled "
            f"{compiled} after warm-up on {sorted(warm.items(), key=str)}"
        )


# inside a DBO microbatch (TP4: prefill steps of >= 8192 tokens, so >= 4096 rows per
# microbatch) the 4-wave rows launch the plain grid, other kernels than outside one
DBO_RANGES = {"fourwave_nt": range(4096, 16385)}


@pytest.mark.parametrize("name", list(DBO_RANGES))
@torch.inference_mode()
def test_dbo_no_compile_after_warmup(name, monkeypatch):
    require_gfx950()
    monkeypatch.setattr(g, "DBO", True)
    _, _, n, k, config = next(case for case in CASES if case[0] == name)
    served = DBO_RANGES[name]
    warm, probe = warm_and_probe(served, lambda rows: variant(name, rows, n, k, config))
    x, _, _, (w, w_scale) = make_inputs(served[-1], n, k)
    for rows in warm.values():
        mxfp8_linear(x[:rows], w, w_scale, config=config)
    torch.cuda.synchronize()

    compiled = count_compiles(monkeypatch)
    for rows in probe:
        mxfp8_linear(x[:rows], w, w_scale, config=config)
        assert not compiled, f"{name} (DBO): M={rows} compiled {compiled} after warm-up"

