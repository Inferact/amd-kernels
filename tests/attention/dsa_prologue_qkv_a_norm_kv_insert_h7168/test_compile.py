# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4): both kernels have one variant per mode (insert or not; top-k
width); the token count, slots and strides are runtime arguments."""

import flydsl.compiler as flyc
import torch
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.indexer_prep import indexer_prep
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.qkv_a_norm_kv_insert import qkv_a_norm_kv_insert

from .cases import EPS, SERVED_M, WEIGHT_SCALE, require_gfx950


@torch.inference_mode()
def test_no_compile_after_warmup(monkeypatch):
    require_gfx950()
    dev = "cuda"
    n_max = SERVED_M[-1]
    x = torch.randn((n_max, 2112), device=dev).bfloat16()
    qw, kw = torch.ones(1536, device=dev).bfloat16(), torch.ones(512, device=dev).bfloat16()
    kvc = torch.zeros((8, 64, 656), dtype=torch.uint8, device=dev)
    slots = torch.arange(n_max, dtype=torch.int64, device=dev)
    q, kv = torch.empty((n_max, 1536), device=dev).bfloat16(), torch.empty((n_max, 512), device=dev).bfloat16()
    iq = torch.randn((n_max, 32, 128), device=dev).bfloat16()
    ik = torch.randn((n_max, 128), device=dev).bfloat16()
    iw = torch.randn((n_max, 32), device=dev).bfloat16()
    lw, lb = torch.ones(128, device=dev).bfloat16(), torch.zeros(128, device=dev).bfloat16()
    ic = torch.zeros((8, 64, 68), dtype=torch.uint8, device=dev)
    out = torch.zeros((n_max + 8, 2048), dtype=torch.int32, device=dev)
    qp = torch.empty((n_max, 32, 64), dtype=torch.uint8, device=dev)
    qs = torch.empty((n_max, 1, 4, 16, 4), dtype=torch.uint8, device=dev)
    wo = torch.empty((n_max, 32), dtype=torch.float32, device=dev)

    def run(n):
        qkv_a_norm_kv_insert(x[:n], qw, kw, EPS, q[:n], kv[:n], kvc, slots[:n])
        qkv_a_norm_kv_insert(x[:n], qw, kw, EPS, q[:n], kv[:n])
        indexer_prep(iq[:n], ik[:n], iw[:n], lw, lb, EPS, slots[:n], ic, out[: n + 8], WEIGHT_SCALE, qp[:n], qs[:n], wo[:n])

    run(1)
    torch.cuda.synchronize()
    compiled = []
    real = flyc.compile
    monkeypatch.setattr(flyc, "compile", lambda fn, *a, **k: (compiled.append(getattr(fn, "kernel_name", "")), real(fn, *a, **k))[1])
    for n in SERVED_M:
        run(n)
        assert not compiled, f"{n} rows compiled {compiled}"
