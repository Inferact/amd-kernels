# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Every decode row the N 4992 host serves keeps its operands in VGPRs: the decode kernel parks an operand in an
AGPR when its registers run out and the copy back races the AGPR write (``mxfp8_gemm_decode.VGPR_BUDGET``: wrong
sums now and then). At N 4992 the deepest lookahead parks on several tiles (e.g. 192x48 three or four K-waves), so
the final ISA of each served row must contain no ``v_accvgpr_write``. Compiled in a subprocess, compile-only, cache
off, ISA dumped."""

import glob
import os
import subprocess
import sys
from pathlib import Path

import torch

from ..mxfp8_common.test_gemm import require_gfx950


def _rows():
    """The decode kernel's rows (the split-K decode rows: tests/gemm/mxfp8_common/test_decode_splitk.py)."""
    from amd_kernels.gemm.mxfp8_common.gemm import _DECODE_KERNEL
    from amd_kernels.gemm.mxfp8_n4992_k6144 import host as H

    return sorted({c for _, c in H.DECODE_ROWS + H.DEFERRED_ROWS if c.kernel == _DECODE_KERNEL}, key=str)


def test_decode_rows_hold_operands_in_vgprs(tmp_path):
    require_gfx950()
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, AKR_COMPILE_ONLY="1", FLYDSL_DEBUG_DUMP_ASM="1", FLYDSL_DUMP_DIR=str(tmp_path),
               FLYDSL_RUNTIME_ENABLE_CACHE="0")
    subprocess.run([sys.executable, "-m", "tests.gemm.mxfp8_n4992_k6144.test_isa"], cwd=root, env=env, check=True)
    names = [g.kernel_name for g in _launchers()]
    assert len(names) == len(_rows())
    for name in names:
        isa = glob.glob(f"{tmp_path}/{name}/*.s")
        assert isa, name
        assert "v_accvgpr_write" not in Path(isa[0]).read_text(), name


def _launchers():
    from amd_kernels.gemm.mxfp8_common import gemm as g

    return [g._decode_launcher(4992, 6144, config, False) for config in _rows()]


if __name__ == "__main__":
    from amd_kernels.common.tensor_shim import _run_compiled

    d = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    for config, exe in zip(_rows(), _launchers()):
        _run_compiled(exe, d, d, d, d, d, config.tile_m, 1024, 1, config.split_k * (4992 // config.tile_n),
                      torch.cuda.current_stream())
