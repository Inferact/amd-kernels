# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The split-K exchange of the kernel assumes that workgroups b and b' with b = b' mod 8 run on the same XCD."""
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T

BLOCKS = 2048


def compile_probe():
    @flyc.kernel(name="xcc_probe", known_block_size=[64, 1, 1])
    def kernel(arg_out: fx.Int64):
        bid = fx.block_idx.x
        xcc = fx.Int32(_llvm.inline_asm(T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)", "=s",
                                         has_side_effects=True))
        out = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=BLOCKS * 4)
        buffer_ops.buffer_store(xcc, out, bid, mask=fx.thread_idx.x == 0)

    @flyc.jit
    def launch(arg_out: fx.Int64, stream: fx.Stream):
        kernel(arg_out).launch(grid=(BLOCKS, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def test_workgroup_xcd_is_bid_mod_8():
    out = torch.full((BLOCKS,), -1, dtype=torch.int32, device="cuda")
    exe = compile_probe()
    flyc.compile(exe, out.data_ptr(), torch.cuda.current_stream())
    torch.cuda.synchronize()
    xcc = out.view(BLOCKS // 8, 8).cpu()
    assert bool((xcc >= 0).all())
    assert bool((xcc == xcc[0:1]).all()), xcc[:4]
