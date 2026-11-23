# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``reference.py`` against vLLM's own Triton ``_hash_ids_kernel`` and ``_engram_lookup_kernel`` (and its
``EngramLayout`` / ``compute_hash_multipliers`` for the tables): the hash columns and the dequantized rows are bit for
bit the same on served-like decode batches. Skipped without vLLM's DeepSeek-V4.1 module."""

import pytest
import torch
from amd_kernels.engram.ngram_rows_mxfp8_d256_c24_ranks8 import reference as ref

from . import workload as wl

engram = pytest.importorskip("vllm.models.deepseek_v41.common.engram")
BUCKET = 4099


def vllm_layout():
    class Config:
        engram_layer_ids = wl.LAYERS
        engram_num_embeddings = (1, 1)
        engram_max_ngram_size = wl.MAX_NGRAM
        engram_vocab_size = BUCKET
        engram_n_heads = wl.HEADS
        engram_head_dim = 256
        engram_compressed_vocab_size = wl.COMPRESSED_VOCAB
        engram_pad_token_id = wl.PAD_ID_TOKEN
        quantization_config = {}

    return engram.EngramLayout(Config)


def test_tables_equal_vllm():
    primes, offsets, mult = wl.layout(BUCKET)
    lay = vllm_layout()
    assert torch.equal(primes, torch.tensor(lay.primes).flatten(1))
    assert torch.equal(offsets, lay.offsets.long())
    assert torch.equal(mult, engram.compute_hash_multipliers(wl.LAYERS, wl.MAX_NGRAM, wl.COMPRESSED_VOCAB))


@pytest.mark.parametrize("requests,rows,pad,image_every", [(16, 1, 16, 0), (5, [1, 6, 3], 32, 0), (32, 6, 256, 7),
                                                             (64, 6, 384, 0), (128, 6, 768, 11)])
def test_hash_and_rows_equal_vllm(requests, rows, pad, image_every):
    from vllm.triton_utils import triton

    primes, offsets, mult = wl.layout(BUCKET)
    tmap = wl.token_map(3)
    pad_id = int(tmap[wl.PAD_ID_TOKEN])
    ids, pos, qsl, lookback, total = wl.decode_batch(requests, rows, pad, 7 + pad, image_every=image_every,
                                                     slots=requests + 3)
    totals = [int(primes[layer].sum()) for layer in range(2)]
    tables = [wl.table(n, 20 + layer) for layer, n in enumerate(totals)]
    dev = torch.device("cuda")
    ids_d, pos_d, qsl_d, lb_d = ids.to(dev), pos.to(dev), qsl.to(dev), lookback.to(dev)
    dead = ids_d == wl.IMAGE_ID
    lb_dead = lb_d == wl.IMAGE_ID
    out = torch.empty(pad, 2, 24, dtype=torch.int32, device=dev)
    engram._hash_ids_kernel[(triton.cdiv(pad, 32), 2)](
        ids_d, tmap.to(dev), dead, pos_d, qsl_d, mult.to(dev), primes.to(dev).view(2, 3, 8), offsets.to(dev).view(2, 3, 8),
        lb_d, lb_dead, out, pad, pad_id, ids_d.stride(0), dead.stride(0), pos_d.stride(0), qsl_d.stride(0),
        qsl_d.numel() - 1, MAX_NGRAM=4, num_heads=8, BLOCK_T=32, BLOCK_H=8, dead_id=-1, lookback_depth=3,
        lookback_row_stride=lb_d.stride(0), lookback_col_stride=lb_d.stride(1), lookback_mask_row_stride=lb_dead.stride(0),
        lookback_mask_col_stride=lb_dead.stride(1), num_warps=4)
    weights = [t[0].to(dev) for t in tables]
    scales = [t[1].to(dev) for t in tables]
    index, codes, shuf = ref.ngram_rows_reference(
        ids_d, pos_d, qsl_d, lb_d, tmap.to(dev), mult.to(dev), primes.to(dev), offsets.to(dev), pad_id, weights,
        scales, 0, image_id=wl.IMAGE_ID)
    assert torch.equal(index.permute(1, 0, 2).to(torch.int32), out)
    for layer in range(2):
        rows = torch.empty(pad, 24, 256, dtype=torch.bfloat16, device=dev)
        engram._engram_lookup_kernel[(64,)](
            weights[layer].view(torch.float8_e4m3fn), scales[layer], out[:, layer], out, rows, 0, totals[layer],
            pad * 24, out.stride(0), 1, HEAD_START=0, LOCAL_HEADS=24, TOTAL_HEADS=24,
            DIM=256, QUANT_BLOCK=32, BLOCK_R=16, GRID=64, SORTED=False, PACKED=False, OUT_STRIDE=256, num_warps=4)
        idx = out[:, layer].long()
        sc = scales[layer][idx.flatten()].view(pad, 192)
        mine = ref.dequant_reference(codes[layer], sc)
        assert torch.equal(mine.view(torch.int16), rows.view(torch.int16))
