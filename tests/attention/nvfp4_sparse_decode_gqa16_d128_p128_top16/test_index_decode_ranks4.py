# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode block on four GPUs (``IndexDecodeTop16`` with a four-rank group, the replicated index cache): each
rank's ids and attention output bit for bit the one-GPU three-launch chain on its own index head, on both sides of the
request-count threshold (one-GPU ``fused`` below, ``split4`` from ``SPLIT_MIN_REQS``), in graph replay at other
lengths too. A CI target of its own with four cards."""
import pytest
import torch

from .test_sparse_decode import require_gfx950


def _worker(rank, port, errfile):
    try:
        _run(rank, port)
    except Exception:
        import traceback

        with open(f"{errfile}.{rank}", "w") as f:
            f.write(traceback.format_exc())
        raise


def _run(rank, port):
    import torch.distributed as dist
    from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import index_decode as blk
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import decode_top16
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.decode_top16 import IndexDecodeTop16

    from .test_index_decode import three_launch_chain
    from .test_index_select import MAX, build

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4)
    for compute, ql in ((torch.bfloat16, 8), (torch.bfloat16, 4), (torch.float16, 8), (torch.float16, 4)):
        index = IndexDecodeTop16(dev, query_len=ql, max_reqs=64, max_seq_len=MAX, rank=rank, group=dist.group.WORLD)
        assert index.split is not None, index.split_off
        for reqs in (1, 2, 4, 5, 6, 7, 8, 17, 64):
            c = build([max(1, 131072 - 3001 * r) for r in range(reqs)], ql, seed=reqs * 13 + ql)
            g = torch.Generator(device="cuda").manual_seed(reqs + 100 * ql)
            iq4 = (torch.randn(c["rows"], 4, 128, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
            c["iq"] = iq4[:, rank:rank + 1].contiguous()
            rows = c["rows"]
            want_ids = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
            want = torch.empty(rows, 16, 128, dtype=torch.bfloat16, device="cuda")
            three_launch_chain(c, 2, 3, want_ids, want, None, compute)
            ids = torch.full_like(want_ids, -7)
            out = torch.empty_like(want)

            def step():
                return blk.nvfp4_index_sparse_decode(
                    index, c["iq"], c["icache"], c["bt"], c["seq"], 2, 3, c["q"], c["kv"], ids, c["bt"], c["seq"], 128**-0.5, out,
                    k_scale=None, v_scale=None, decode_query_len=ql, max_seq_len=MAX, compute_type=compute)

            r = step()
            torch.cuda.synchronize()
            assert r == ("split4" if reqs >= decode_top16.SPLIT_MIN_REQS[ql] else "fused"), (ql, reqs, r)
            assert torch.equal(ids, want_ids), (rank, compute, ql, reqs, r)
            assert torch.equal(out.view(torch.int16), want.view(torch.int16)), (rank, compute, ql, reqs, r)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            c["seq"].copy_(torch.tensor([max(1, 70000 - 999 * i) for i in range(reqs)], dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            three_launch_chain(c, 2, 3, want_ids, want, None, compute)
            torch.cuda.synchronize()
            assert torch.equal(ids, want_ids) and torch.equal(out.view(torch.int16), want.view(torch.int16)), (
                rank, ql, reqs, "replay")
    dist.destroy_process_group()


def test_index_decode_four_gpus(tmp_path):
    require_gfx950()
    if torch.cuda.device_count() < 4:
        pytest.skip("needs four GPUs")
    import os

    import torch.multiprocessing as mp

    from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host  # noqa: F401

    from ...indexer.indexer_nvfp4_d128_p128_top16.test_split4_rank_owned import _free_port

    errfile = str(tmp_path / "err")
    try:
        mp.spawn(_worker, args=(_free_port(), errfile), nprocs=4, join=True)
    except Exception:
        msgs = [open(f"{errfile}.{r}").read() for r in range(4) if os.path.exists(f"{errfile}.{r}")]
        raise AssertionError("\n".join(msgs) or "worker failed") from None
