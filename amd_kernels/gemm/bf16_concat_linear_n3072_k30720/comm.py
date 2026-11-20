# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eager two-rank N-shard projection with a two-slot shard exchange."""
import ctypes
import functools
import torch
import torch.distributed as dist
from amd_kernels.common import hip_runtime as ops
from . import host
from .host import run
from .reduce_gather import DATA_OFF,GROUPS,compile_reduce_gather

WORLD=2
# (max rows, rows per group, column pieces per row, push (write the peer's slot) or pull (read it)); sweep:
# benchmarks/gemm/bf16_concat_linear_n3072_k30720/benchmark_comm.py --sweep
EXCHANGE=((32,1,6,False),(1<<31,1,1,False))


class _DevicePointer:
    """A raw device allocation as a tensor (``__cuda_array_interface__``)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def exchange(obj, rank: int, group) -> list:
    """Every rank's ``obj`` (broadcasts on the CPU group: all_gather_object fails
    on gloo under inference mode)."""
    out = [[None] for _ in range(WORLD)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


@functools.cache
def reduce_gather_kernel(rpw,cpg,split,push):return compile_reduce_gather(rpw,cpg,split,push)


def _fit(rows,rpw,cpg):
    while (rows+rpw-1)//rpw*cpg>GROUPS:rpw*=2
    return rpw


def exchange_for(rows):
    """(rows per group, column pieces, push) of the shard exchange; groups never exceed the flag lines."""
    rpw,cpg,push=next((r,c,p) for top,r,c,p in EXCHANGE if rows<=top)
    return _fit(rows,rpw,cpg),cpg,push


class ConcatLinearComm:
    def __init__(self,packed_weight,rank,group,max_rows=host.MAX_ROWS):
        assert dist.get_world_size(group)==WORLD and packed_weight.shape==(host.N,host.K) and 1<=max_rows<=host.MAX_ROWS
        self.weight,self.rank,self.group=packed_weight,rank,group
        self.max_rows=max_rows
        self.slot_bytes=max_rows*host.N*2
        ptr,handle=ops.allocate_shared_buffer_and_handle(DATA_OFF+2*self.slot_bytes)
        every=exchange(handle,rank,group)
        self.bufs=[ptr if r==rank else ops.open_mem_handle(every[r]) for r in range(WORLD)]
        self.counts=torch.zeros(GROUPS,dtype=torch.int32,device=packed_weight.device)
        self.call=0
        torch.cuda.synchronize();dist.barrier(group=group)

    def __call__(self,taps,out,config=None,exchange_config=None):
        """The fc of the five taps into ``out`` [rows, 6144]; ``config`` / ``exchange_config`` override the tier and
        (rows per group, column pieces, push) (sweeps)."""
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('full concat-linear communicator is eager only; Python slot rotation cannot replay')
        m=taps[0].shape[0]
        assert m<=self.max_rows and out.shape==(m,2*host.N) and out.is_contiguous() and out.dtype==torch.bfloat16
        slot=self.call%2;self.call+=1
        cfg=config or host.config_for(m)
        partials=host._partials_config(taps,self.weight,*cfg)
        rpw,cpg,push=exchange_config or exchange_for(m)
        run(reduce_gather_kernel(_fit(m,rpw,cpg),cpg,cfg[3],push),partials,*self.bufs,self.counts.data_ptr(),out,m,
            self.rank,slot,self.slot_bytes,torch.cuda.current_stream())
        return out

    def warm_up(self):
        """Compile every (tier, exchange) variant a row count up to ``max_rows`` reaches."""
        reps={}
        for m in range(1,self.max_rows+1):
            reps.setdefault((host.config_for(m),exchange_for(m)),m)
        for m in reps.values():
            xs=[torch.zeros((m,host.H),dtype=torch.bfloat16,device=self.weight.device) for _ in range(5)]
            self(xs,torch.empty((m,2*host.N),dtype=torch.bfloat16,device=self.weight.device))

    def close(self):
        torch.cuda.synchronize();dist.barrier(group=self.group)
        hip=ops._hip()
        hip.hipFree.argtypes=[ctypes.c_void_p];hip.hipIpcCloseMemHandle.argtypes=[ctypes.c_void_p]
        for r,p in enumerate(self.bufs):
            rc=(hip.hipFree if r==self.rank else hip.hipIpcCloseMemHandle)(ctypes.c_void_p(p))
            assert rc==0,rc
        self.bufs=[]
