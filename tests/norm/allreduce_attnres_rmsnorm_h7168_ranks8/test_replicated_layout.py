# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The all-gather as a sum: every rank holds a [M, 7168] partial that is zero outside its own 896 columns (the
'replicated layout' of the latent up-projection kernel). The all-reduce of such partials must equal, bit for bit,
the stock layout: the all-gathered tensor fed to the same kernel with ``reduce=False``. Zero padding is exact except for
the sign of zero: a real -0.0 beside seven +0.0 sums to +0.0; -0.0 padding keeps every bit. Both paddings are
checked: -0.0 padding must match bit for bit, +0.0 padding must match everywhere except the sign of zeros."""
import torch
import torch.distributed as dist
from amd_kernels.norm.allreduce_attnres_rmsnorm_h7168_ranks8 import AttnResComm

from tests.norm.allreduce_attnres_rmsnorm_h7168_ranks8.multi_gpu8 import run_on_8_gpus
from tests.norm.allreduce_attnres_rmsnorm_h7168_ranks8.workload import SERVED_ROWS, make

SHARD = 896


def bits(t):
    return t.contiguous().view(torch.uint8)


def worker(rank, errors):
    dev = torch.device("cuda", rank)
    c = AttnResComm(rank, dist.group.WORLD, dev)
    for m in SERVED_ROWS:
        parts, prefix0, bank0, nw, qk, ow = make(m, 9100 + m, dev)
        gathered = torch.cat([p[:, r * SHARD:(r + 1) * SHARD] for r, p in enumerate(parts)], dim=1).contiguous()
        gathered[:, ::97] = -0.0  # real negative zeros in every shard
        gathered[:, 5::89] = 0.0
        for nb, widx, hp in ((3, -1, True), (4, 4, True), (0, 0, False)):
            prefix, bank = (prefix0.clone() if hp else None), bank0.clone()
            ref, ref_p = c.allreduce_attnres_rmsnorm(gathered, prefix, bank, nw, qk, nb, widx, ow, reduce=False)
            ref_bank = bank.clone()
            for pad, exact in ((-0.0, True), (0.0, False)):
                mine = torch.full((m, 7168), pad, dtype=torch.bfloat16, device=dev)
                mine[:, rank * SHARD:(rank + 1) * SHARD] = gathered[:, rank * SHARD:(rank + 1) * SHARD]
                prefix, bank = (prefix0.clone() if hp else None), bank0.clone()
                out, pout = c.allreduce_attnres_rmsnorm(mine, prefix, bank, nw, qk, nb, widx, ow)
                torch.cuda.synchronize()
                tag = (rank, m, nb, widx, pad)
                if exact:
                    if not (torch.equal(bits(out), bits(ref)) and torch.equal(bits(pout), bits(ref_p))
                            and torch.equal(bits(bank), bits(ref_bank))):
                        errors.append(f"not bit exact {tag}")
                else:
                    if not (torch.equal(out, ref) and torch.equal(pout, ref_p) and torch.equal(bank, ref_bank)):
                        errors.append(f"values differ {tag}")
    c.close()


def test_replicated_layout_equals_all_gather():
    run_on_8_gpus(worker, timeout=600)
