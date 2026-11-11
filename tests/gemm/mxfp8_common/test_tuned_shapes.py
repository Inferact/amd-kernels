# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``gemm.tuned_shapes``: the (N, K) shapes with dtype-specific rows, the set a caller switches by shape."""
import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as G

# MiniMax-M3.1 TP4: qkv, o_proj, dense gate_up, dense down, shared gate_up, shared down, DSpark draft qkv
TP4 = {(2560, 6144), (6144, 2048), (6144, 6144), (6144, 3072), (1536, 6144), (6144, 768), (2304, 6144)}


def test_bf16_is_the_tables_bf16_keys():
    table = G._load_tuned_configs()
    got = G.tuned_shapes(torch.bfloat16)
    assert got == {(n, k) for n, k, d in table if d == "bf16"}
    assert G.tuned_shapes() == got
    assert TP4 <= got


def test_any_only_shapes_are_out():
    table = G._load_tuned_configs()
    any_only = {(n, k) for n, k, d in table if d == "any"} - {(n, k) for n, k, d in table if d == "bf16"}
    assert not any_only & G.tuned_shapes(torch.bfloat16)


def test_fp16_and_other_dtypes():
    table = G._load_tuned_configs()
    assert G.tuned_shapes(torch.float16) == {(n, k) for n, k, d in table if d == "fp16"}
    with pytest.raises(KeyError):
        G.tuned_shapes(torch.float32)


def test_every_tuned_shape_selects_a_bf16_row_at_m1():
    """At one row each tuned shape picks a row of its own bf16 block, not an AITER default."""
    table = G._load_tuned_configs()
    for n, k in G.tuned_shapes(torch.bfloat16):
        rows = table[(n, k, "bf16")]
        if rows[0][1].min_m > 1:
            continue
        assert G._select_config(1, n, k) in [c for _, c in rows], (n, k)


@pytest.mark.parametrize("n,k", sorted(TP4))
@torch.inference_mode()
def test_weight_warmup_covers_every_m(n, k, monkeypatch):
    """After ``warmup_mxfp8_gemm_weight`` over 320 rows, the bf16 entry at any M up to 320 compiles nothing
    (no FlyDSL kernel, the activation quantizer included)."""
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels.gemm.mxfp8_common.reference import quantize_mxfp8

    from .test_gemm import require_gfx950

    require_gfx950()
    torch.manual_seed(n + k)
    w, ws = G.preshuffle_mxfp8_weight(*quantize_mxfp8(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05))
    G.warmup_mxfp8_gemm_weight(w, ws, 320, torch.bfloat16)
    flydsl_compiled = []
    real = MlirCompiler.compile

    def counting(self, *a, **kw):
        flydsl_compiled.append(1)
        return real(self, *a, **kw)

    monkeypatch.setattr(MlirCompiler, "compile", counting)
    x = torch.zeros((320, k), device="cuda", dtype=torch.bfloat16)
    for m in (1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 47, 64, 65, 100, 127, 128, 129, 191, 192, 255, 256, 257, 300, 320):
        G.mxfp8_linear(x[:m], w, ws, torch.bfloat16)
    torch.cuda.synchronize()
    assert not flydsl_compiled, len(flydsl_compiled)
