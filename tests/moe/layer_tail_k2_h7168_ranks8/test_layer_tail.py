# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The layer tail on 8 ranks against ``reference`` (cosine >= 0.9999 on the kernel's own AttnRes output, bit-exact
across ranks and replays), run as one torchrun job over 8 GPUs."""
import os
import pathlib
import subprocess
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_layer_tail_matches_reference_on_8_ranks():
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "8", "--master_port",
           str(29500 + os.getpid() % 1000), "-m", "tests.moe.layer_tail_k2_h7168_ranks8.worker"]  # fmt: skip
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert proc.stdout.strip().splitlines()[-1] == "OK", proc.stdout[-3000:]
