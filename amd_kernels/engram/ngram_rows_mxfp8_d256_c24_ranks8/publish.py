# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Publish GPU-initialized immutable tables once, before peer lookup starts.

A system-scope release fence on eight workgroups per XCD writes back every XCD's dirty L2 lines
(``buffer_wbl2 sc0 sc1``), so the peers' reads see the tables.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from amd_kernels.common import buffer_ops

# Eight groups per XCD, as in pull_collectives_ranks4's startup fence.
GROUPS = 64


def compile_table_publish():
    @flyc.kernel(name="ngram_table_publish_d256", known_block_size=[64, 1, 1])
    def kernel(done: fx.Tensor):
        _llvm.FenceOp(_llvm.AtomicOrdering.release, syncscope="")
        if fx.thread_idx.x == 0:
            dst = buffer_ops.create_buffer_resource(done, num_records_bytes=GROUPS * 4)
            buffer_ops.buffer_store(fx.Int32(1), dst, fx.block_idx.x, cache_modifier=0x11)

    @flyc.jit
    def launch_table_publish(done: fx.Tensor, stream: fx.Stream):
        kernel(done).launch(grid=(GROUPS, 1, 1), block=(64, 1, 1), stream=stream)
    launch_table_publish.kernel_name = "ngram_table_publish_d256"
    return launch_table_publish
