# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""On 8 ranks: the layer tail runner from the entry; ``layer_tail_launch`` (a build mapping with extra keys, the
runner's own launch arguments) is bit for bit the package's ``k2_launch`` on every rank at 4 and 16 rows, reset on and
off; a call the runner was not built for gets a reason; after the first launch of each build nothing compiles, and a
graph of the entry launch, captured with ``importlib.import_module`` unavailable, replays the eager result."""
import importlib
from unittest import mock

import torch
import torch.distributed as dist

from tests.moe.layer_tail_k2_h7168_ranks8.inputs import layer_inputs, moe_weights
from tests.multi_gpu8 import run_on_8_gpus

KEY = (7168, 1536, 896, 8)
LAYER = 131


def bits(t):
    return t.contiguous().view(torch.uint8)


def body(rank, errors):
    from amd_kernels import compile_watch
    from amd_kernels.moe import latent_moe_common as E
    from amd_kernels.moe.latent_moe_common import layer_tail

    dev = torch.device("cuda", rank)
    group = dist.group.WORLD
    host = importlib.import_module(layer_tail._SHAPE_HOSTS[KEY])
    dims = dict(hidden=7168, o_proj_k=1536, experts=896)
    if E.layer_tail_runner_unsupported_reason(dev, group, rank, **dims) is not None:
        errors.append(f"rank {rank}: factory reason")
    runner = E.layer_tail_runner(dev, group, rank, **dims)
    _, wd = moe_weights(rank, dev)

    def args(s, seed=0):
        w = layer_inputs(rank, dev, s, seed=seed)
        kw = dict(core=w["core"], w_o=w["w_o"], blocks=w["blocks"], ares_nw=w["ares_nw"], ares_qk=w["ares_qk"],
                  in_nw=w["in_nw"], **wd, layer=LAYER)
        return w["prefix"], kw

    for s in (4, 16):
        for reset in (False, True):
            prefix, kw = args(s)
            build = {"tokens": s, "nblocks": 4 + reset, "reset": reset, "m1b_qlen": 0, "dsa": False}
            pa, pb = prefix.clone(), prefix.clone()
            oa, ob = (torch.empty(s, 7168, dtype=torch.bfloat16, device=dev) for _ in range(2))
            if E.layer_tail_launch_unsupported_reason(runner, build, prefix=pb, out=ob, **kw) is not None:
                errors.append(f"rank {rank}: reason on served rows {s}")
            runner.step_begin()
            host.k2_launch(host.K2Build(tokens=s, nblocks=4 + reset, reset=reset), prefix=pa, out=oa, **kw,
                           **runner.launch_args(), **runner.peer_args())
            runner.step_begin()
            E.layer_tail_launch(runner, build, prefix=pb, out=ob, **kw)
            torch.cuda.synchronize()
            if not (torch.equal(bits(oa), bits(ob)) and torch.equal(bits(pa), bits(pb))):
                errors.append(f"rank {rank}: rows {s} reset {reset} differ from the package launch")

    prefix, kw = args(4)
    out = torch.empty(4, 7168, dtype=torch.bfloat16, device=dev)
    other = torch.empty(384, 768, 1792, dtype=torch.uint8, device="meta")
    reason = E.layer_tail_launch_unsupported_reason(runner, {"tokens": 4, "nblocks": 4}, prefix=prefix, out=out,
                                                    **{**kw, "w13": other})
    if "runner built for hidden=7168, o_proj_k=1536, experts=896, world=8, call is" not in (reason or ""):
        errors.append(f"rank {rank}: mismatch reason {reason!r}")

    msgs = []
    compile_watch.arm(msgs.append)
    late0 = compile_watch.late_compiles()
    prefix, kw = args(16, seed=1)
    build = {"tokens": 16, "nblocks": 5, "reset": True}
    p, eager = prefix.clone(), torch.empty(16, 7168, dtype=torch.bfloat16, device=dev)
    runner.step_begin()
    E.layer_tail_launch(runner, build, prefix=p, out=eager, **kw)
    eager_prefix = p.clone()
    out = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), mock.patch("importlib.import_module", side_effect=RuntimeError("import")):
        runner.step_begin()
        E.layer_tail_launch(runner, build, prefix=p, out=out, **kw)
    p.copy_(prefix)
    graph.replay()
    torch.cuda.synchronize()
    if not (torch.equal(bits(out), bits(eager)) and torch.equal(bits(p), bits(eager_prefix))):
        errors.append(f"rank {rank}: graph replay differs from eager")
    if compile_watch.late_compiles() != late0:
        errors.append(f"rank {rank}: compiles after the first launch: {msgs[:2]}")
    if runner.health.any() or runner.flags.any():
        errors.append(f"rank {rank}: a wait gave up")
    del graph
    compile_watch.disarm()
    runner.close()


def test_layer_tail_entry_on_8_ranks():
    run_on_8_gpus(body, timeout=900)
