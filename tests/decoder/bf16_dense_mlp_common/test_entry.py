# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The dense MLP entry (workspace, packing, op) gives the routed package's bits at the smallest and the largest served
row count; its reasons are None on those calls."""
import importlib

import pytest
import torch

from amd_kernels.decoder.bf16_dense_mlp_common import (bf16_dense_mlp, bf16_dense_mlp_shapes,
                                                       bf16_dense_mlp_unsupported_reason, dense_mlp_pack_down,
                                                       dense_mlp_pack_gate_up, dense_mlp_packed_tiles,
                                                       dense_mlp_rows_buffer, dense_mlp_workspace,
                                                       dense_mlp_workspace_unsupported_reason, host as entry)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("key", sorted(bf16_dense_mlp_shapes()))
def test_entry_equals_package(key):
    hidden, inter = key
    host = importlib.import_module(entry._SHAPE_HOSTS[key])
    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(hidden + inter)
    gu = (torch.randn(2 * inter, hidden, device=dev, generator=g) * 0.02).bfloat16()
    dn = (torch.randn(hidden, inter, device=dev, generator=g) * 0.03).bfloat16()
    assert dense_mlp_workspace_unsupported_reason(dev, hidden=hidden, inter=inter) is None
    ws_e, ws_d = dense_mlp_workspace(dev, hidden=hidden, inter=inter), host.MlpWorkspace(dev)
    gu_e, dn_e = dense_mlp_pack_gate_up(gu), dense_mlp_pack_down(dn)
    gu_d, dn_d = host.pack_gate_up(gu), host.pack_down(dn)
    assert torch.equal(gu_e, gu_d) and torch.equal(dn_e, dn_d)
    for rows in (1, host.MAX_ROWS):
        x = (torch.randn(rows, hidden, device=dev, generator=g) * 0.5).bfloat16()
        tiles = dense_mlp_packed_tiles(rows, down=dn_e)
        assert tiles == host.packed_tiles(rows)
        xp = dense_mlp_rows_buffer(rows, dev, down=dn_e)
        xp.copy_(host.pack_rows(x, tiles))
        out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=dev)
        assert bf16_dense_mlp_unsupported_reason(ws_e, xp, gu_e, dn_e, rows, out) is None
        ref = host.dense_mlp(ws_d, xp.clone(), gu_d, dn_d, rows)
        got = bf16_dense_mlp(ws_e, xp, gu_e, dn_e, rows, out)
        torch.cuda.synchronize()
        assert got.data_ptr() == out.data_ptr()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), rows
