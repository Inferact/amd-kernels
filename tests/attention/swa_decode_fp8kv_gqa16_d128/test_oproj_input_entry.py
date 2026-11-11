# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``swa_decode_fp8kv_oproj_input`` (the attention as its o_proj's input, the MXFP8 / bf16 choice and the codes'
buffers in the library): bit for bit what the kernel entry writes when the caller makes the same choice itself
(the TP4 DSpark 3 / 7 shapes, 1-64 requests); its first call of a mode compiles every request count's variant."""
import json
import os
import subprocess
import sys

import pytest
import torch
from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128 import swa_decode_fp8kv, swa_decode_fp8kv_oproj_input
from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128.oproj_input import mxfp8_out
from amd_kernels.gemm.mxfp8_common import gemm as G

from .test_decode import SM_SCALE, W, make_batch, require_gfx950, scales
from .test_oproj_input import K, N, code_buffers, real_bytes


@pytest.mark.parametrize("ql", [3, 7])
@pytest.mark.parametrize("reqs", [1, 2, 5, 11, 16, 48, 64])
@pytest.mark.parametrize("causal", [False, True])
def test_entry_equals_kernel_calls(ql, reqs, causal):
    require_gfx950()
    gen = torch.Generator().manual_seed(reqs * 10 + ql)
    pool, bt, sl, qsl, q, _ = [t.cuda() if isinstance(t, torch.Tensor) else t
                               for t in make_batch(reqs, ql, 0, gen, 128, 1)]
    ks, vs = scales()
    rows = q.shape[0]
    args = (pool, bt, sl, qsl, ql, SM_SCALE, W, causal, ks, vs)
    # bf16: the entry without an o_proj (3-D and 2-D q)
    want = torch.full_like(q, float("nan"))
    swa_decode_fp8kv(q, *args, want)
    for qq in (q, q.view(rows, -1)):
        got = swa_decode_fp8kv_oproj_input(qq, *args)
        assert got.shape == qq.shape and torch.equal(got.view(torch.int16), want.view(qq.shape).view(torch.int16))
    # with the o_proj: MXFP8 codes from 8 rows of a 3 / 7-row block, else bf16
    got = swa_decode_fp8kv_oproj_input(q.view(rows, -1), *args, oproj_n=N)
    if not mxfp8_out(rows, ql):
        assert isinstance(got, torch.Tensor) and torch.equal(got.view(torch.int16), want.view(rows, -1).view(torch.int16))
        return
    codes, sc, tile = got
    assert tile == G.prequant_code_tile_rows(rows, N, K, torch.bfloat16)
    want_c, want_s = code_buffers(rows, tile)
    swa_decode_fp8kv(q, *args, None, quant_out=(want_c, want_s, tile))
    torch.cuda.synchronize()
    for a, b in zip(real_bytes(codes, sc, rows, tile), real_bytes(want_c, want_s, rows, tile)):
        assert torch.equal(a, b)


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    ks, vs = scales()
    bad = []
    for reqs in (1, 2, 3, 5, 6, 10, 11, 17, 33, 48, 49, 64, 100, 128, 200, 256):
        for ql in (3, 7):
            gen = torch.Generator().manual_seed(reqs)
            pool, bt, sl, qsl, q, _ = [t.cuda() if isinstance(t, torch.Tensor) else t
                                       for t in make_batch(reqs, ql, 0, gen, 128, 1, ctx_hi=600)]
            before = len(late)
            for oproj in (None, N):
                swa_decode_fp8kv_oproj_input(q.view(q.shape[0], -1), pool, bt, sl, qsl, ql, SM_SCALE, W, False, ks,
                                             vs, oproj_n=oproj)
            if len(late) > before:
                bad.append((reqs, ql))
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1500,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0 and result["warmed"] == 4, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
