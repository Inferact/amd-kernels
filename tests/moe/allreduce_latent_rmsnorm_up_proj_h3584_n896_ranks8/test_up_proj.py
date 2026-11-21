# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The up-projection role on one GPU (``mode="up"``: it reads ``xn`` and the squared sums from the ``lat`` buffer): every
served row count, fresh inputs on every run (a stale latent read shows), an output that aliases the shared view, and
bit-identical reruns."""
import pytest
import torch
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8 import tail_kernel as T
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.ipc import f32_bits, run_compiled
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.reference import up_proj_reference

from .workload import K, N, rand

ROWS = (1, 12, 16, 24, 40, 65, 80, 128, 192, 256)


def run(m, runs=3):
    dev = torch.device("cuda")
    lat_buf = torch.zeros(T.LAT_ALLOC // 2, dtype=torch.bfloat16, device=dev)
    xn_rows = lat_buf[:T.MAX_ROWS * K].view(T.MAX_ROWS, K)
    ssq = lat_buf[T.MAX_ROWS * K: T.MAX_ROWS * K + 2 * T.MAX_ROWS * T.WORLD].view(torch.float32).view(T.MAX_ROWS, T.WORLD)
    table = torch.tensor([0] * 8 + [lat_buf.data_ptr()] * 8, dtype=torch.int64, device=dev)
    counts = torch.zeros(T.CNT_WORDS, dtype=torch.int32, device=dev)
    groups = T.groups_of((m + 15) // 16)
    mpg = T.mpg_of((m + 15) // 16, groups)
    kernel = T.compile_tail("up", mpg, T.rb_of(m), False, groups, T.nsub_of((m + 15) // 16))
    for r in range(runs):
        x = rand((T.MAX_ROWS, K), 10 * r + m, dev, 1.5)
        nw = (1 + rand((K,), 10 * r + 1, dev, 0.1)).bfloat16()
        w = rand((N, K), 10 * r + 2, dev, 0.03)
        xn_rows.copy_((x.float() * nw.float()).bfloat16())
        ssq.copy_((x.float() ** 2).view(T.MAX_ROWS, T.WORLD, K // T.WORLD).sum(-1))
        full = rand((m, 7168), 10 * r + 3, dev)
        view = full[:, 896:1792]
        shared0 = view.clone()

        def call():
            run_compiled(kernel, m, (m + T.rb_of(m) - 1) // T.rb_of(m), 0, table.data_ptr(), counts.data_ptr(), 0, 0, nw.data_ptr(),
                         w.data_ptr(), view.data_ptr(), view.stride(0), view.data_ptr(), view.stride(0),
                         f32_bits(1e-6), 0, 0, 0, torch.cuda.current_stream())
            torch.cuda.synchronize()

        call()
        y, up = up_proj_reference(x[:m], nw, w, shared0)
        err = (view.double() - y).abs()
        tol = 2.0**-8 * y.abs() + 2.0**-8 * up.abs().amax(-1, keepdim=True) + 1e-6
        assert bool((err <= tol).all()), (m, r, float(err.max()))
        cos = torch.nn.functional.cosine_similarity(view.double(), y, dim=1)
        assert float(cos.min()) >= 0.99999, (m, r, float(cos.min()))
        again = view.clone()
        view.copy_(shared0)
        call()
        assert torch.equal(view.view(torch.int16), again.view(torch.int16)), ("rerun", m)


@pytest.mark.parametrize("m", ROWS)
def test_up_proj(m):
    run(m)
