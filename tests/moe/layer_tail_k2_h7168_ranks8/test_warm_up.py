# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``layer_tail`` is an entry (``amd_kernels.warm_up``): its first call of a mode (nblocks, reset) compiles every
served width, launching nothing (the runner's scratch, health and epoch stay zero); calls at the served widths afterwards compile nothing. Compile-only: no kernel launches."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    import torch.distributed as dist
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.moe.layer_tail_k2_h7168_ranks8 import SERVED_ROWS, LayerTailRunner, LayerTailWeights, layer_tail

    from tests.moe.layer_tail_k2_h7168_ranks8.inputs import layer_inputs, moe_weights

    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:%d" % (29600 + os.getpid() % 300), rank=0, world_size=1)
    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    device = torch.device("cuda")
    comm = LayerTailRunner(device, dist.group.WORLD, 0)
    _, dev = moe_weights(0, device)
    bad = []
    for nblocks, reset in ((4, False), (5, True)):
        for n in (*SERVED_ROWS, 6, 12):
            w = layer_inputs(0, device, n)
            weights = LayerTailWeights(w["w_o"], w["ares_nw"], w["ares_qk"], w["in_nw"], **dev)
            before = len(late)
            layer_tail(comm, weights, w["core"], w["prefix"], w["blocks"], nblocks=nblocks, reset=reset, layer=3)
            if len(late) > before:
                bad.append((nblocks, reset, n))
    torch.cuda.synchronize()
    touched = bool(comm.scratch.any().item() or comm.health.any().item() or comm.epoch.item())
    return {"late": bad, "warmed": len(warm_up.warmed()), "touched": touched}


def test_first_call_warm_up_compiles_every_served_width():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1500,
                          env={**os.environ, "COMPILE_ONLY": "1"})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["warmed"] == 2 and not result["touched"], result


if __name__ == "__main__":
    print(json.dumps(scenario()))
