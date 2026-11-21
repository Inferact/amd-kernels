# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after the first call (root README, rule 4): the kernel's compile key is the tokens per sequence, the
norm weight dtype, the snapshot mode and uniform lengths; the sequence count, the row count and the slot pitch are
runtime."""

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.attention.kda_decode_fb_gate_spec_h12_d128 import kda_decode_fb_gate_spec

from .inputs import make_inputs
from .test_decode import require_gfx950

SERVED = [12, 24, 40, 65, 80, 128, 192]


@torch.inference_mode()
@pytest.mark.parametrize("norm_f32", [False, True])
def test_no_compile_after_warmup(norm_f32, monkeypatch):
    require_gfx950()
    kda_decode_fb_gate_spec(**make_inputs(SERVED[0], norm_f32=norm_f32))
    torch.cuda.synchronize()
    compiled = []
    real = flyc.compile

    def counting(fn, *args, **kwargs):
        compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting)
    for rows in SERVED[1:]:
        kda_decode_fb_gate_spec(**make_inputs(rows, norm_f32=norm_f32, pad_seqs=rows % 3,
                                              slot_pitch=12 * 128 * 128 + 64 * (rows % 2)))
        assert not compiled, f"{rows} rows compiled {compiled}"
