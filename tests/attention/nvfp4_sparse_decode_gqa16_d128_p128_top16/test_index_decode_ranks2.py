# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode block on two GPUs (``IndexDecodeTop16`` with a two-rank group, the replicated index cache, TP2: 2 index /
KV heads per rank): each rank's ids and attention output (bf16 or MXFP8) bit for bit the one-GPU three-launch chain
on its own heads, on both sides of the request-count threshold (one-GPU ``fused`` below, ``split2`` from
``SPLIT2_MIN_REQS``), in graph replay at other lengths too."""
import torch

from ...multi_gpu import run_on_2_gpus
from .test_index_decode import three_launch_chain
from .test_index_select import MAX, build
from .test_sparse_decode import quant_buffers, require_gfx950


def _body(rank, errors):
    import torch.distributed as dist
    from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import index_decode as blk
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import decode_top16
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.decode_top16 import IndexDecodeTop16

    dev = torch.device("cuda", rank)
    for compute, ql in ((torch.float16, 8), (torch.float16, 4), (torch.bfloat16, 8)):
        index = IndexDecodeTop16(dev, query_len=ql, max_reqs=64, max_seq_len=MAX, num_index_heads=2, rank=rank,
                                 group=dist.group.WORLD, warm_up=False)
        m = decode_top16.SPLIT2_MIN_REQS[ql]
        for reqs in (1, m - 1, m, 17, 64):
            for quant in (False, True):
                c = build([max(1, 131072 - 3001 * r) for r in range(reqs)], ql, seed=reqs * 13 + ql, hkv=2)
                g = torch.Generator(device="cuda").manual_seed(reqs + 100 * ql)
                iq4 = (torch.randn(c["rows"], 4, 128, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
                c["iq"] = iq4[:, 2 * rank : 2 * rank + 2].contiguous()
                rows = c["rows"]
                want_ids = torch.full((2, rows, 16), -7, dtype=torch.int32, device="cuda")
                want = torch.full((rows, 32, 128), 7.0, dtype=torch.bfloat16, device="cuda")
                wq = (*quant_buffers(rows, 4096), 16) if quant else None
                three_launch_chain(c, 2, 3, want_ids, want, wq, compute)
                ids = torch.full_like(want_ids, -7)
                out = torch.full_like(want, 7.0)
                q = (*quant_buffers(rows, 4096), 16) if quant else None

                def step():
                    return blk.nvfp4_index_sparse_decode(
                        index, c["iq"], c["icache"], c["bt"], c["seq"], 2, 3, c["q"], c["kv"], ids, c["bt"],
                        c["seq"], 128**-0.5, out, k_scale=None, v_scale=None, decode_query_len=ql, max_seq_len=MAX,
                        quant_out=q, compute_type=compute)

                def same():
                    ok = torch.equal(ids, want_ids)
                    if quant:
                        return ok and torch.equal(q[0], wq[0]) and torch.equal(q[1], wq[1])
                    return ok and torch.equal(out.view(torch.int16), want.view(torch.int16))

                r = step()
                torch.cuda.synchronize()
                tag = (rank, compute, ql, reqs, quant, r)
                if r != ("split2" if reqs >= m else "fused"):
                    errors.append(f"{tag}: route")
                if not same():
                    errors.append(f"{tag}: ids / output differ")
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    step()
                c["seq"].copy_(torch.tensor([max(1, 70000 - 999 * i) for i in range(reqs)], dtype=torch.int32))
                graph.replay()
                torch.cuda.synchronize()
                three_launch_chain(c, 2, 3, want_ids, want, wq, compute)
                torch.cuda.synchronize()
                if not same():
                    errors.append(f"{tag}: replay ids / output differ")


def test_index_decode_two_gpus():
    require_gfx950()
    run_on_2_gpus(_body)
