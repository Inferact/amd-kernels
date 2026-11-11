# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The table selection (``gemm._select_config``) at the decode windows and the TP4 rows.

These pin the window edges the deployments rely on: the producer code tiles (``prequant_code_tile_rows``),
the split-K rows below the windows, the rows above.
"""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_n6144_k8192 import host as H_n6144_k8192

BF16 = torch.bfloat16


def kernel(m, n, k, dtype=BF16, allow_decode=True):
    return g._select_config(m, n, k, dtype, allow_decode).kernel


def _decode_window(n, k, lo, incumbent_below, inc384_waves_m):
    """The o_proj / down decode window (lo < M <= 416): at M <= 384 the incumbent M = 384
    row's 96x96x128 split-K-1 tile with four K-waves and sched_barriers on; at 385..416
    the one-wave 112x96 tile (a fifth 96-row M tile would push the grid past 256 CTAs;
    RT 7 through the decode kernel's odd-RT A-scale re-pairing). Without allow_decode
    the "any" AITER row."""
    for m in range(lo + 1, 417):
        config = g._select_config(m, n, k, BF16)
        tile = 96 if m <= 384 else 112
        assert config.kernel == "decode", f"M={m}: {config.kernel}"
        assert (config.tile_m, config.tile_n, config.tile_k) == (tile, 96, 128), m
        assert config.split_k == 1
        assert (config.waves_m, config.waves_n) == (4, 1), m  # kw4, sched_barriers on
        assert g.prequant_code_tile_rows(m, n, k, BF16) == tile
        raw = g._select_config(m, n, k, BF16, False)
        assert (raw.kernel, raw.waves_m) == ("aiter", 0), f"M={m} allow_decode=False"
    below = g._select_config(lo, n, k, BF16)
    assert (below.kernel, below.tile_m, below.tile_n, below.waves_m, below.waves_n) == (
        "decode",
        *incumbent_below,
    )
    # above the window: the AITER row, or for o_proj (6144, 8192) the split-2 256x192 row of
    # test_tp4_prefill_rows
    assert kernel(417, n, k) == ("prefill_n6144_k8192_256x192_split2" if (n, k) == (6144, 8192) else "aiter")
    # the incumbent M <= 384 row itself (shadowed inside the window) is unchanged
    rows = g._load_tuned_configs()[(n, k, "bf16")]
    inc384 = next(c for m_, c in rows if m_ == 384)
    assert (inc384.tile_m, inc384.tile_n, inc384.waves_m, inc384.waves_n) == (
        96,
        96,
        inc384_waves_m,
        0,
    )


def test_o_proj_k8192_window():
    """(6144, 8192): the window is 288 < M <= 416; M = 288 keeps its 96x64 kw2 row. The
    attention epilogue's shared code buffer takes tiles <= 128 at every decode M."""
    _decode_window(6144, 8192, 288, (96, 64, 2, 1), 2)
    assert all(g.prequant_code_tile_rows(m, 6144, 8192, BF16) <= 128 for m in range(1, 417))


@pytest.mark.parametrize(
    "n,k,m,name",
    [
        (6144, 8192, 320, "flydsl_mxfp8_dec_n6144_k8192_rt6_ni6_kw4_ks1_la1x6sb"),
    ],
)
def test_decode_window_kernel_objects(n, k, m, name):
    """The windows dispatch other kernel objects than the incumbent rows (a timing of
    the window would otherwise silently measure the incumbent), and the incumbent M <= 384
    row's own object name is unchanged (flags 0)."""
    pytest.importorskip("flydsl")
    ours = g._decode_launcher(n, k, g._select_config(m, n, k, BF16), False)
    assert ours.kernel_name == name, ours.kernel_name
    ours416 = g._decode_launcher(n, k, g._select_config(416, n, k, BF16), False)
    assert "_rt7_ni6_kw4_ks1_" in ours416.kernel_name, ours416.kernel_name
    assert ours416.kernel_name.endswith("sb"), ours416.kernel_name
    rows = g._load_tuned_configs()[(n, k, "bf16")]
    inc384 = next(c for m_, c in rows if m_ == 384)
    inc_name = g._decode_launcher(n, k, inc384, False).kernel_name
    kw = inc384.waves_m
    assert inc_name == f"flydsl_mxfp8_dec_n{n}_k{k}_rt6_ni6_kw{kw}_ks1_la1x6", inc_name
    assert ours.kernel_name != inc_name


def test_tp4_prefill_rows():
    """The TP4 rows (graph replay A/B on layer 39 rank 0's weights, bitwise-equal outputs):
    QKV (2560, 6144) at 385-1536 (but 513-576) decode rows with 80-column tiles (eight
    row tiles x 32 column tiles = one workgroup per CU); 1537-2048 uses the rectangular
    128x160 tile, 2049-3072 4-wave NT 128x256, 3073-10240 4-wave NT 256x256; o_proj
    (6144, 2048) the rectangular 128x192 tile at 769-1024, 4-wave NT 256x128 at 1025-1280,
    4-wave NT 256x256 at 1281-3328, the persistent 256x192 tile at 3329-3968, the
    AITER row kept at 3969-4096 (no stable eager gain there), 4-wave NT 256x256 at 4097-12288; o_proj_full (6144, 8192, the SP o_proj) the
    split-2 256x192 tile at 417-1024, 4-wave NT 128x256 at 1025-1280, 4-wave NT 256x256 at
    1281-4608, the 4-wave row from 4609 (NT stores). The decode window below
    417 keeps its decode rows and every allow_decode=False row at <= 416 the AITER row."""

    def tile(m, n, k, allow_decode=True):
        c = g._select_config(m, n, k, BF16, allow_decode)
        return c.kernel, c.tile_m, c.tile_n

    assert tile(384, 2560, 6144) == ("decode", 64, 64)
    bn80 = {(385, 512): 64, (577, 640): 80, (641, 768): 96, (769, 896): 112, (897, 1024): 128, (1025, 1152): 144,
            (1153, 1280): 160, (1281, 1408): 176, (1409, 1536): 192}
    for (lo, hi), tm in bn80.items():
        for m in (lo, (lo + hi) // 2, hi):
            assert tile(m, 2560, 6144) == ("decode", tm, 80), m
            # 6-12 m-blocks re-read each W column: cached W loads
            assert g._select_config(m, 2560, 6144, BF16).w_cm == 0, m
            assert g._select_config(m, 2560, 6144, BF16).waves_m == (4 if tm == 80 else 3), m
            assert g.prequant_code_tile_rows(m, 2560, 6144, BF16) == tm, m
            # a row-major-code caller gets the AITER default row there
            assert tile(m, 2560, 6144, False)[0] == "aiter", m
    for m in (513, 576):
        assert tile(m, 2560, 6144) == ("decode", 96, 64), m
        assert g._select_config(m, 2560, 6144, BF16).w_cm is None, m
    # deferred TP4 QKV partials: cached W loads at 193-288 rows only
    for m, wcm in ((128, None), (129, None), (176, None), (192, None), (193, 0), (256, 0), (257, 0), (288, 0)):
        d = g._deferred_qkv_config(m, 2560, 6144, g._select_prequant_config(m, 2560, 6144, BF16))
        assert d.w_cm == wcm, (m, d)
    for m in (1537, 1600, 2048):
        assert tile(m, 2560, 6144) == ("rect_n2560_k6144", 128, 160), m
        assert g.prequant_code_tile_rows(m, 2560, 6144, BF16) == 0
    assert tile(2049, 2560, 6144) == ("4wave_nt", 128, 256)
    assert tile(3072, 2560, 6144) == ("4wave_nt", 128, 256)
    for m in (3073, 3584, 4095, 4096, 4097, 4608, 4799, 4800):
        assert tile(m, 2560, 6144) == ("prefill_n2560_k6144_192x256", 192, 256), m
        assert tile(m, 2560, 6144, False) == ("prefill_n2560_k6144_192x256", 192, 256), m
    for m in (4801, 6144, 8192, 8193, 10240):
        assert tile(m, 2560, 6144) == ("4wave_nt", 256, 256), m
    assert tile(10241, 2560, 6144) == ("4wave", 256, 256)
    # o_proj decode rows sized to their rows (same m-block count); 161-192 / 257-320 four
    # K-waves with the park-all K-reduce
    for lo, hi, want in ((33, 48, (48, 32, 1)), (49, 64, (32, 48, 1)), (65, 80, (80, 32, 1)), (81, 96, (96, 32, 1)), (129, 160, (80, 48, 4)), (161, 192, (96, 48, 4)), (257, 320, (80, 96, 4)),
                         (321, 384, (96, 96, 2)), (385, 448, (112, 96, 2)), (449, 512, (128, 96, 2))):
        for m in (lo, hi):
            c = g._select_prequant_config(m, 6144, 2048, BF16)
            assert (c.kernel, c.tile_m, c.tile_n, c.waves_m) == ("decode", *want), (m, c)
            assert g.prequant_code_tile_rows(m, 6144, 2048, BF16) == want[0], m
    assert tile(768, 6144, 2048)[0] != "n6144_k2048_128x192"
    for allow in (True, False):
        for m in (769, 800, 897, 1024):
            assert tile(m, 6144, 2048, allow) == ("n6144_k2048_128x192", 128, 192), m
            assert g.prequant_code_tile_rows(m, 6144, 2048, BF16) == 0, m
    assert tile(1025, 6144, 2048) == ("4wave_nt", 256, 128)
    assert tile(1280, 6144, 2048) == ("4wave_nt", 256, 128)
    for m in (1281, 2048, 2049, 3072):
        assert tile(m, 6144, 2048) == ("4wave_nt", 256, 256), m
    for m in (3073, 3200, 3328):
        assert tile(m, 6144, 2048) == ("4wave_nt", 256, 256), m
    for m in (3329, 3584, 3968):
        for allow in (True, False):
            assert tile(m, 6144, 2048, allow) == ("prefill_n6144_k8192_256x192", 256, 192), m
        assert g.prequant_code_tile_rows(m, 6144, 2048, BF16) == 0, m
    for m in (3969, 4095, 4096):
        assert tile(m, 6144, 2048) == ("aiter", 128, 256), m
    for m in (4097, 8191, 12288):
        assert tile(m, 6144, 2048) == ("4wave_nt", 256, 256), m
    assert tile(12289, 6144, 2048) == ("4wave", 256, 256)
    for allow in (True, False):
        for m in (1025, 1280):
            assert tile(m, 6144, 8192, allow) == ("4wave_nt", 128, 256), m
        for m in (417, 512, 513, 768, 769, 896, 897, 960, 1024):
            assert tile(m, 6144, 8192, allow) == ("prefill_n6144_k8192_256x192_split2", 256, 192), m
        for m in (1281, 1535, 1536, 1537, 2048):
            want = "prefill_n6144_k8192_256x192_one_tile" if m <= 1536 else "prefill_n6144_k8192_256x192"
            assert tile(m, 6144, 8192, allow) == (want, 256, 192), m
        for m in (2049, 4095):
            assert tile(m, 6144, 8192, allow) == ("4wave_nt", 256, 256), m
        for m in (4096, 4097, 4608):
            assert tile(m, 6144, 8192, allow) == ("4wave_nt", 256, 256), m
        assert tile(4609, 6144, 8192, allow) == ("4wave", 256, 256)
    for m in (289, 385, 416):
        assert tile(m, 6144, 8192, False)[0] == "aiter", m
    # rows below the window keep the AITER row
    assert tile(384, 6144, 8192, False)[0] == "aiter"


def test_one_tile_grid_guard():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='at most 256'):
        H_n6144_k8192._launch_prefill_n6144_k8192_192(None, None, None, None, SimpleNamespace(shape=(2049, 6144)), one_tile=True)


@pytest.mark.parametrize("m", [1, 8, 15, 16, 33, 129, 416, 1752])
def test_prequant_buffers(m):
    """o_proj's pre-quantized input buffers: rows padded to ``prequant_code_tile_rows``' tile (16 for row-major),
    scales padded to 32 rows."""
    n, k = 6144, 2048
    codes, scales, tile = g.prequant_buffers(m, n, k, torch.device("cpu"))
    pad = max(tile, 16)
    assert tile == g.prequant_code_tile_rows(m, n, k, BF16)
    assert codes.shape == (-(-m // pad) * pad, k) and codes.dtype == torch.uint8
    assert scales.shape == (-(-codes.shape[0] // 32) * 32, k // 32) and scales.dtype == torch.uint8
