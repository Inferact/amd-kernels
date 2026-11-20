# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two ranks: sum this rank's fp32 K slabs into its bf16 column shard and exchange the shards into [rows, 6144].

A group is ``rpw`` rows x one of ``cpg`` column pieces of the 3072-column shard: the fixed-order slab sum, rounded
once to bf16, goes into the local output and into a staging slot (``push``: the peer's slot over xGMI; otherwise this
rank's own slot, which the peer then pulls); a system-scope release, one epoch flag on the peer's line of the group, a
spin on this rank's line; then the peer's piece comes from the slot (``push``: the local slot; otherwise the peer's)
into the output."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec
from .flags import _flag_store,_spin

C=3072  # columns per rank
LINE=32
GROUPS=1024
DATA_OFF=LINE*GROUPS*4
ROW_BYTES=C*2
SYS=0x11


def compile_reduce_gather(rpw,cpg,split,push):
    R=rpw
    T=C//8//cpg
    assert T*8*cpg==C and T%32==0,cpg
    name=f"bf16_reduce_gather_ranks2_n6144_c{C}_r{R}_p{cpg}_sk{split}_{'push' if push else 'pull'}"
    @flyc.kernel(name=name,known_block_size=[T,1,1])
    def kernel(partials:fx.Tensor,b0:fx.Int64,b1:fx.Int64,
               counts:fx.Int64,out:fx.Tensor,rows:fx.Int32,me:fx.Int32,
               slot:fx.Int32,slot_bytes:fx.Int64):
        tx,g=fx.thread_idx.x,fx.block_idx.x
        mine=(me==0).select(b0,b1)
        theirs=(me==0).select(b1,b0)
        peer=1-me
        rb,piece=g//cpg,g%cpg
        col=piece*(T*8)+tx*8
        cr=buffer_ops.create_buffer_resource_from_addr(counts,num_records_bytes=GROUPS*4)
        epoch=fx.Int32(buffer_ops.buffer_load(cr,g,vec_width=1,dtype=fx.Int32,is_scalar=True))+1
        pr=buffer_ops.create_buffer_resource(partials,num_records_bytes=fx.Int64(rows)*(C*split*4))
        dst=buffer_ops.create_buffer_resource(out,num_records_bytes=fx.Int64(rows)*(2*C*2))
        stage_out=theirs if push else mine
        stage_in=mine if push else theirs
        for r in range(fx.Int32(0),fx.Int32(R),fx.Int32(1)):
            row=rb*R+r
            live=row<rows
            sums=[fx.Float32(0.0) for _ in range_constexpr(8)]
            for slab in range_constexpr(split):
                a=Vec(buffer_ops.buffer_load(pr,(slab*rows+row)*C+col,vec_width=4,dtype=fx.Float32,mask=live))
                b=Vec(buffer_ops.buffer_load(pr,(slab*rows+row)*C+col+4,vec_width=4,dtype=fx.Float32,mask=live))
                sums=[sums[j]+fx.Float32(a[j] if j<4 else b[j-4]) for j in range_constexpr(8)]
            packed=Vec.from_elements([fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(sums[2*j]),fx.as_ir_value(sums[2*j+1]))) for j in range_constexpr(4)],fx.Int32)
            sr=buffer_ops.create_buffer_resource_from_addr(stage_out+DATA_OFF+fx.Int64(slot)*slot_bytes+fx.Int64(row)*ROW_BYTES,num_records_bytes=ROW_BYTES)
            buffer_ops.buffer_store(packed,sr,col//2,mask=live,cache_modifier=SYS)
            buffer_ops.buffer_store(packed,dst,row*C+(me*C+col)//2,mask=live)
        # Every wave's staging stores complete before the barrier; the release then orders them before the flag.
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        if tx==0:
            _llvm.FenceOp(_llvm.AtomicOrdering.release,syncscope="")
            _flag_store(theirs+fx.Int64((g*LINE+me)*4),epoch)
            _spin(mine+fx.Int64((g*LINE+peer)*4),epoch)
        fx.gpu.barrier()
        for r in range(fx.Int32(0),fx.Int32(R),fx.Int32(1)):
            row=rb*R+r
            live=row<rows
            rr=buffer_ops.create_buffer_resource_from_addr(stage_in+DATA_OFF+fx.Int64(slot)*slot_bytes+fx.Int64(row)*ROW_BYTES,num_records_bytes=ROW_BYTES)
            v=Vec(buffer_ops.buffer_load(rr,col//2,vec_width=4,dtype=fx.Int32,mask=live,cache_modifier=SYS))
            buffer_ops.buffer_store(v,dst,row*C+(peer*C+col)//2,mask=live)
        buffer_ops.buffer_store(epoch,cr,g,mask=tx==0)
    @flyc.jit
    def launch_bf16_reduce_gather_ranks2_n6144(partials:fx.Tensor,b0:fx.Int64,b1:fx.Int64,
                                     counts:fx.Int64,out:fx.Tensor,rows:fx.Int32,me:fx.Int32,
                                     slot:fx.Int32,slot_bytes:fx.Int64,stream:fx.Stream):
        kernel(partials,b0,b1,counts,out,rows,me,slot,slot_bytes).launch(
            grid=(fx.Int64((rows+R-1)//R*cpg),1,1),block=(T,1,1),stream=stream)
    launch_bf16_reduce_gather_ranks2_n6144.kernel_name=name
    return launch_bf16_reduce_gather_ranks2_n6144
