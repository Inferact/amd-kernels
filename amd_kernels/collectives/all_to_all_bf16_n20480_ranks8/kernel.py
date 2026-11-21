# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Push exchange of logits rows: the sender writes into the receivers' final layout, the receiver only waits.

Protocol (shared with the GEMM's push epilogue, `gemm/bf16_n20480_k7168`): every rank owns BUFFERS receive buffers
[RECV_ROWS, 163840] bf16 and a 512-byte uncached flag block, all mapped into every peer (HIP IPC; the exchange's
`peers` table: the eight bases of buffer 0, the eight of buffer 1, then the eight flag blocks). A sender stores its 20480 columns of a row at
byte `row * 327680 + rank * 40960` of the owner's buffer with system-scope stores, waits for them, and the last of its
workgroups (a two-level ticket: per XCD, then across the eight XCDs, on the sender's own flag block) stores the
sequence number into word `96 + rank` of every receiver's flag block. `wait` spins on the eight words of its own
block (local memory) until each is >= its device sequence counter, so a receiver reads its rows in place. The
sequence counter is a device word that the last arriver advances; no host state enters a graph. The caller picks the
buffer of each exchange (a launch constant, so a graph keeps its addresses) and alternates two buffers between
consecutive exchanges: a sender pushes into buffer b of exchange e + 2 only after it saw every rank's flag of exchange
e + 1, and a rank publishes that flag only after its own consumers of buffer b (stream order before its next push)
finished, so no receive buffer is overwritten while it is read. The rows of an exchange are valid until the next
exchange on the same buffer is pushed (the next but one of the communicator).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

N = 20480
WORLD = 8
V = WORLD * N
RECV_ROWS = 256
BUFFERS = 2
FLAGS = BUFFERS * WORLD  # index of the first flag block in the peers table
NT = 256
SYS = 0x11
FLAG_WORD = 96  # first word of the incoming flags in a flag block
OOB = 0x3FFFFFF0


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def flag_add(addr_i64, v):
    """Atomic add at system scope on an uncached word; returns the old value."""
    return fx.Int32(
        _llvm.AtomicRMWOp(_llvm.AtomicBinOp.add, _global_i32_ptr(addr_i64).llvm_ptr, fx.Int32(v).ir_value(),
                          _llvm.AtomicOrdering.monotonic, syncscope="", alignment=4).result)


def compile_wait():
    @flyc.kernel(name="all_to_all_bf16_n20480_ranks8_wait", known_block_size=[64, 1, 1])
    def kernel(counter: fx.Tensor, arg_flag: fx.Int64):
        tx = fx.thread_idx.x
        seq = fx.Int32(counter[0])
        if tx < WORLD:
            addr = arg_flag + fx.Int64((FLAG_WORD + tx) * 4)
            val = flag_add(addr, 0)
            # modular int32 comparison stays valid across counter wrap
            while (val - seq) < 0:
                val = flag_add(addr, 0)
        fx.gpu.barrier()
        _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="")

    @flyc.jit
    def launch_all_to_all_bf16_n20480_ranks8_wait(counter: fx.Tensor, arg_flag: fx.Int64, stream: fx.Stream):
        kernel(counter, arg_flag).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    return launch_all_to_all_bf16_n20480_ranks8_wait


def compile_push(mode):
    """Rows [rows, 20480] of this rank to their owners (``owner``: rows sorted by owner, counts per owner) or to every
    rank (``all``): half a row per workgroup, the grid padded to a multiple of 8 workgroups (the XCD tickets)."""
    assert mode in ("owner", "all")
    PASSES = N // 2 // 8 // NT  # 16-byte vectors per thread per half row: 10240 / 8 / 256 = 5

    @flyc.kernel(name=f"all_to_all_bf16_n20480_ranks8_push_{mode}", known_block_size=[NT, 1, 1])
    def kernel(arg_x: fx.Int64, peers: fx.Tensor, counts: fx.Tensor, counter: fx.Tensor, arg_flag: fx.Int64,
               i32_rows: fx.Int32, i32_rank: fx.Int32, i32_nwg: fx.Int32, i32_buf: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        row = bid // 2
        half = bid % 2
        src = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_rows * (N * 2))
        dst = [buffer_ops.create_buffer_resource_from_addr(fx.Int64(peers[i32_buf * WORLD + d]),
                                                           num_records_bytes=RECV_ROWS * V * 2)
               for d in range_constexpr(WORLD)]
        cums = [fx.Int32(0)]
        for j in range_constexpr(WORLD):
            cums.append(cums[j] + fx.Int32(counts[j]))
        owner = fx.Int32(0)
        lo = fx.Int32(0)
        for j in range_constexpr(1, WORLD):
            ge = row >= cums[j]
            owner = ge.select(fx.Int32(j), owner)
            lo = ge.select(cums[j], lo)
        live = row < i32_rows
        drow = (row - lo) if mode == "owner" else row
        for p in range_constexpr(PASSES):
            col = half * (N // 2) + (p * NT + tx) * 8  # the bf16 element of this thread's 16 bytes
            v = Vec(buffer_ops.buffer_load(src, (row * N + col) // 2, vec_width=4, dtype=fx.Int32, mask=live))
            off = (drow * V + i32_rank * N + col) // 2
            for d in range_constexpr(WORLD):
                ok = (live & (owner == d)) if mode == "owner" else live
                buffer_ops.buffer_store(v, dst[d], ok.select(off, fx.Int32(OOB)), cache_modifier=SYS)
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        if tx == 0:
            xw = arg_flag + fx.Int64((bid % 8) * 32 + 64)
            t1 = flag_add(xw, 1)
            if t1 == i32_nwg // 8 - 1:
                flag_add(xw, -(i32_nwg // 8))
                t2 = flag_add(arg_flag + 4, 1)
                if t2 == WORLD - 1:
                    flag_add(arg_flag + 4, -WORLD)
                    seq = fx.Int32(counter[0]) + 1
                    counter[0] = seq
                    for d in range_constexpr(WORLD):
                        fr = buffer_ops.create_buffer_resource_from_addr(fx.Int64(peers[FLAGS + d]),
                                                                       num_records_bytes=512)
                        buffer_ops.buffer_store(seq, fr, FLAG_WORD + i32_rank, cache_modifier=SYS)

    @flyc.jit
    def launch_push(arg_x: fx.Int64, peers: fx.Tensor, counts: fx.Tensor, counter: fx.Tensor, arg_flag: fx.Int64,
                    i32_rows: fx.Int32, i32_rank: fx.Int32, i32_nwg: fx.Int32, i32_buf: fx.Int32, stream: fx.Stream):
        kernel(arg_x, peers, counts, counter, arg_flag, i32_rows, i32_rank, i32_nwg, i32_buf).launch(
            grid=(i32_nwg, 1, 1), block=(NT, 1, 1), stream=stream)
    return launch_push
