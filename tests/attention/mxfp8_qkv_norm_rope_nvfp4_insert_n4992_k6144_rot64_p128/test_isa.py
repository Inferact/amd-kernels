# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Every GEMM this entry compiles itself (the one-launch kernel at 1-16 rows, the non-temporal-store decode rows at
257-448) keeps its MFMA operands in VGPRs: a parked operand's copy back races the AGPR write (``gemm/mxfp8_common``
``mxfp8_gemm_decode.VGPR_BUDGET``), so the final ISA holds no ``v_accvgpr_write``. Compiled in a subprocess,
compile-only, cache off, ISA dumped."""

import glob
import os
import subprocess
import sys
from pathlib import Path

import torch

from .test_qkv_insert import require_gfx950

ROWS = list(range(1, 17)) + list(range(257, 449))


def _launchers():
    from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128 import host

    fused = {host._plan(m) for m in ROWS if m <= host.FUSED_MAX_ROWS}
    nt = {host._nt_config(m) for m in ROWS if m > host.FUSED_MAX_ROWS} - {None}
    return [host._fused_launcher(p) for p in sorted(fused, key=str)], [host._nt_gemm_launcher(c)
                                                                         for c in sorted(nt, key=str)]


def test_operands_stay_in_vgprs(tmp_path):
    require_gfx950()
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, AKR_COMPILE_ONLY="1", FLYDSL_DEBUG_DUMP_ASM="1", FLYDSL_DUMP_DIR=str(tmp_path),
               FLYDSL_RUNTIME_ENABLE_CACHE="0")
    module = "tests.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128.test_isa"
    subprocess.run([sys.executable, "-m", module], cwd=root, env=env, check=True)
    fused, nt = _launchers()
    assert fused and len(nt) == 3
    for exe in fused + nt:
        isa = glob.glob(f"{tmp_path}/{exe.kernel_name}/*.s")
        assert isa, exe.kernel_name
        assert "v_accvgpr_write" not in Path(isa[0]).read_text(), exe.kernel_name


if __name__ == "__main__":
    from amd_kernels.common.tensor_shim import _run_compiled

    d = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    p = d.data_ptr()
    fused, nt = _launchers()
    for exe in fused:
        _run_compiled(exe, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, 8, 1024, 256, 1000, 1000, 1, 0,
                      torch.cuda.current_stream())
    for exe in nt:
        _run_compiled(exe, d, d, d, d, d, 256, 1024, 1, 104, torch.cuda.current_stream())
