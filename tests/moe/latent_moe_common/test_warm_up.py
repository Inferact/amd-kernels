# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Through the entries, the router -> experts chain warms its packages up on the first call of each mode; walking the
served rows afterwards compiles nothing; a CUDA graph of the chain, captured with ``importlib.import_module``
unavailable, replays bit for bit what the eager chain gives."""
import json
import os
import subprocess
import sys
from unittest import mock

import pytest
import torch

ROWS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 65, 96, 128, 160, 192)


def chain(E, x, w, bias, lat, weights, aux, out=None, slots=False):
    E.router_sigmoid_bias_topk(x, w, bias, 1.0, out, aux=aux)
    sort = E.moe_sort_from_aux(aux, weights=weights)
    codes, scales = E.quantize_moe_activations(lat, weights=weights)
    return E.mxfp4_moe_experts(codes, scales, weights, sort, reduce=not slots)


def scenario():
    from amd_kernels import compile_watch, warm_up
    from amd_kernels.moe import latent_moe_common as E
    from amd_kernels.moe.mxfp4_h3584_i3072_e896_top16 import MoeWeights
    from tests.moe.mxfp4_h3584_i3072_e896_top16.workload import make_checkpoint

    warm_up.set_auto(True)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    w = (torch.randn(896, 7168, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    bias = torch.randn(896, device=dev, generator=g) * 0.1
    ck = MoeWeights.from_checkpoint(*make_checkpoint())
    weights = E.moe_weights(ck.w13, ck.s13, ck.w2, ck.s2)

    def inputs(m):
        x = (torch.randn(m, 7168, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        lat = (torch.randn(m, 3584, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        return x, lat, E.router_aux(w, m, dev, zero_moe_buf=False)

    for slots in (False, True):
        x, lat, aux = inputs(1)
        chain(E, x, w, bias, lat, weights, aux, slots=slots)
    warmed = len(warm_up.warmed())
    msgs = []
    compile_watch.arm(msgs.append)
    for m in ROWS:
        x, lat, aux = inputs(m)
        for slots in (False, True):
            chain(E, x, w, bias, lat, weights, aux, slots=slots)
    torch.cuda.synchronize()
    replays = {}
    for m in (12, 192):
        x, lat, aux = inputs(m)
        out = (torch.empty(m, 16, device=dev), torch.empty(m, 16, dtype=torch.int32, device=dev))
        eager = chain(E, x, w, bias, lat, weights, aux, out).clone()
        topk = [t.clone() for t in out]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), mock.patch("importlib.import_module", side_effect=RuntimeError("import")):
            got = chain(E, x, w, bias, lat, weights, aux, out)
        for t in (got, *out):
            t.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        replays[m] = torch.equal(got, eager) and all(torch.equal(a, b) for a, b in zip(out, topk))
    return {"warmed": warmed, "late": compile_watch.late_compiles(), "msgs": msgs[:4], "replays": replays}


def test_warm_up_then_no_compile_and_graph_replay():
    from amd_kernels.moe.latent_moe_common import available

    if not available():
        pytest.skip("needs gfx950 with FlyDSL")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1500,
                          env=dict(os.environ))
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["warmed"] >= 4 and result["late"] == 0 and all(result["replays"].values()), result


if __name__ == "__main__":
    print(json.dumps(scenario()))
