# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The four-GPU tests of ``test_split4_rank_owned``, a four-card CI target of their own (tools/ci/targets.tsv)."""
import os

import pytest
import torch

from .inputs import require_gfx950
from .test_split4_rank_owned import _free_port, _worker


def _prefix_boundary_worker(rank, port):
    import torch.distributed as dist
    from amd_kernels import warm_up
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4.decode_score import MARKER, TK_MAX_ROWS
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4.host import IndexSplit4RankOwned, _DevicePointer
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4.kernel import CNT_SC
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import rank_owned_position
    from .test_split4_rank_owned import _nvfp4_pages, _single_head_scores, _topk_decode_indices
    torch.set_num_threads(1)
    warm_up.set_auto(True)
    warm_up.set_workers(1)
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=4)
    for ql, n in ((4, 32), (4, 33), (8, 17), (8, 33)):
        cp = IndexSplit4RankOwned(rank, dist.group.WORLD, torch.device('cuda', rank), n * ql, 8192, ql)
        cg = torch.Generator(device='cuda').manual_seed(ql * 100 + n)
        cache = _nvfp4_pages(64, cg)
        table = torch.randint(0, 64, (n, 8192), generator=cg, dtype=torch.int32, device='cuda')
        q = (torch.randn(n * ql, 4, 128, generator=cg, device='cuda') * .5).to(torch.float8_e4m3fn)
        iq = q[:, rank].contiguous()
        seq = torch.empty(n, dtype=torch.int32, device='cuda')
        out = torch.empty(1, n * ql, 16, dtype=torch.int32, device='cuda')
        seq.fill_(ql)
        def run():
            return cp.score_topk(iq, cache, table, seq, 2, 3, 8192, out, 1 << 20, sharded=False)
        run()
        torch.cuda.synchronize()
        assert int(cp.counts[CNT_SC]) == 1, (ql, n, int(cp.counts[CNT_SC]))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for pattern in ([32768, 65543, 300001, 1, 0], [0], [129, 3, 1, 0, 1048576]):
            lens = (pattern * ((n + len(pattern) - 1) // len(pattern)))[:n]
            seq.copy_(torch.tensor(lens, dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            expected = _single_head_scores(q, cache, table, seq, ql, 2, 3)[rank:rank + 1]
            want = _topk_decode_indices(expected, seq, ql, False)
            assert torch.equal(out[0], want), (rank, ql, n, pattern, 'ids')
            if n * ql <= TK_MAX_ROWS:
                # the one-launch path's own receive buffer: all MARKER again
                mark = torch.as_tensor(_DevicePointer(cp.mark_ptrs[rank], n * ql * 8192 * 4), device='cuda')
                assert (mark.view(torch.int32) == MARKER).all(), (rank, ql, n, pattern, 'marker reset')
                continue
            score = cp.score_rcv[:n * ql * 8192].view(1, n * ql, 8192)
            for r, length in enumerate(lens):
                nb = (length + 127) // 128
                pos = rank_owned_position(torch.arange(nb, device='cuda'))
                got = score[:, r * ql:(r + 1) * ql, pos]
                ref = expected[:, r * ql:(r + 1) * ql, :nb]
                assert torch.equal(got.view(torch.int32), ref.view(torch.int32)), (rank, ql, n, r, 'score bits')
        assert int(cp.counts[CNT_SC]) == 4, (ql, n, int(cp.counts[CNT_SC]))
    assert warm_up.late_compiles() == 0
    dist.destroy_process_group()


def test_split4_prefix_boundaries_four_gpus():
    """32/33 requests and the small-prefix standalone route, with changed lengths."""
    require_gfx950()
    if torch.cuda.device_count() < 4:
        pytest.skip('needs four GPUs')
    import torch.multiprocessing as mp
    mp.spawn(_prefix_boundary_worker, args=(_free_port(),), nprocs=4, join=True)


def test_split4_rank_owned_four_gpus(tmp_path):
    """Four ranks on four GPUs over HIP IPC (``init_rank_owned``): decode on
    the rank-owned and the replicated cache bitwise the single-GPU path, three
    runs, both DBO sets on two streams at once, graph replays with new
    lengths."""
    require_gfx950()
    if torch.accelerator.device_count() < 4:
        pytest.skip("needs four GPUs (HIP_VISIBLE_DEVICES with four cards)")
    import torch.multiprocessing as mp

    errfile = str(tmp_path / "err")
    try:
        mp.spawn(_worker, args=(_free_port(), errfile), nprocs=4, join=True)
    except Exception:
        msgs = []
        for r in range(4):
            p = f"{errfile}.{r}"
            if os.path.exists(p):
                with open(p) as f:
                    msgs.append(f.read())
        raise AssertionError("\n".join(msgs) or "worker failed") from None
