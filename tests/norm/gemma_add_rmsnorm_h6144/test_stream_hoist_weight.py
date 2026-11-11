# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The weight-hoisted streamed AR tiers and their unhoisted neighbours, through the staged tests' contract harness
(``_worker``: reference, strided inputs, special values, replays, MFMA co-execution, no compile after warm-up)."""
import pytest
from tests.multi_gpu import run_on_4_gpus
from .test_staged_ar_add_rmsnorm import _worker


def test_stream_hoist_boundaries_reference_replay_and_compile():
    # the staged row-major MXFP8 stream (strided inputs take the staged kernels in every mode)
    rows = (1,31,32,512,1024,1025,2048,2049,2304,2560,2816,3072,3073,4095)
    run_on_4_gpus(_worker, rows, True, True, 0, False, False, True, timeout=1800)


@pytest.mark.parametrize('tile,rows', [
    (16,(256,257,288,289,321,448,480,481,512,536,537,640,641,768,769,1024,1280,1281,1536,1537,2048,2049,3072,3073,4095)),
    (0,(256,257,288,321,448,480,481,512,640,641,768,769,896,897,1023,1024,1280,1281,1536,1537,2048,2049,2304,2560,2816,3072,3073,3584,4095)),
])
def test_registered_mxfp8_stream_hoist_reference_replay_and_compile(tile, rows):
    # contiguous inputs: a strided x skips the registered peer reads (the staged kernel); the eager chain is the
    # registered graph's bitwise oracle
    run_on_4_gpus(_worker, rows, True, True, tile, False, True, False, timeout=1800)
