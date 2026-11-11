# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Dense norm's quant rule and arithmetic stay identical in both code layouts."""
import torch
from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import (
    gemma_add_rmsnorm_mxfp8,gemma_add_rmsnorm_mxfp8_row_major,
)
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import (
    untile_codes, unshuffle_scales, add_rmsnorm_reference, compare_mxfp8,
    SCALE_MISMATCH, CODE_MISMATCH,
)


def test_row_major_matches_tiled_every_size():
    w=(torch.randn(6144,device='cuda')*.5).bfloat16()
    for rows in (0,1,7,8,16,31,32,33,128,512,1024,2048,8192):
        x=torch.randn((rows,6152),device='cuda').bfloat16()[:,:6144]
        r=torch.randn((rows,6160),device='cuda').bfloat16()[:,:6144]
        if rows:
            x[:,:32]=0;r[:,:32]=2.**-133
            x[:,32:64]=-0.;r[:,32:64]=-0.
        q,s,res=gemma_add_rmsnorm_mxfp8_row_major(x,r,w,1e-6)
        tq,ts,tr=gemma_add_rmsnorm_mxfp8(x,r,w,1e-6,32)
        assert torch.equal(q.view(torch.uint8),untile_codes(tq,rows))
        assert torch.equal(unshuffle_scales(s,rows),unshuffle_scales(ts,rows))
        assert torch.equal(res.view(torch.uint8),tr.view(torch.uint8))
        if rows:
            y,_=add_rmsnorm_reference(x.cpu(),r.cpu(),w.cpu(),1e-6)
            stats=compare_mxfp8(q.cpu().view(torch.uint8),unshuffle_scales(s,rows).cpu(),y)
            assert stats['scale_max_diff']<=1 and stats['scale_mismatch']<=SCALE_MISMATCH,stats
            assert stats['code_far']==0 and stats['code_mismatch']<=CODE_MISMATCH,stats


def test_size_consistency():
    """The first 8 rows' outputs do not depend on the row count (bf16 norm, dense row-major codes)."""
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import gemma_add_rmsnorm
    w=torch.randn(6144,device='cuda').bfloat16()
    fixed_x=torch.randn(8,6144,device='cuda').bfloat16()
    fixed_r=torch.randn_like(fixed_x)
    fixed={}
    for rows in (8,16,31,32,33,512,8192):
        x=torch.randn((rows,6152),device='cuda').bfloat16()[:,:6144]
        r=torch.randn((rows,6160),device='cuda').bfloat16()[:,:6144]
        x[:8].copy_(fixed_x);r[:8].copy_(fixed_r)
        ry,rr=gemma_add_rmsnorm(x,r,w,1e-6)
        dq,ds,_=gemma_add_rmsnorm_mxfp8_row_major(x,r,w,1e-6)
        for tag,t in (('y',ry[:8].view(torch.uint8)),('res',rr[:8].view(torch.uint8)),
                      ('dense_q',dq[:8].view(torch.uint8)),('dense_s',unshuffle_scales(ds,rows)[:8])):
            if tag in fixed:assert torch.equal(t,fixed[tag])
            else:fixed[tag]=t.clone()


def test_odd_input_row_strides_against_fp64():
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import gemma_add_rmsnorm
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import compare_out,OUT_COS,OUT_MISMATCH
    rows=33
    w=torch.randn(6144,device='cuda').bfloat16()
    for pad in (1,3,17):
        x=torch.randn(rows,6144+pad,device='cuda').bfloat16()[:,:6144]
        r=torch.randn(rows,6146+pad,device='cuda').bfloat16()[:,:6144]
        y,res=gemma_add_rmsnorm(x,r,w,1e-6)
        ref,rr=add_rmsnorm_reference(x.cpu(),r.cpu(),w.cpu(),1e-6)
        st=compare_out(y.cpu(),ref)
        assert st['bad_ulp']==0 and st['cos']>=OUT_COS and st['mismatch']<=OUT_MISMATCH,st
        assert torch.equal(res.cpu().view(torch.uint8),rr.view(torch.uint8))
        q,s,res=gemma_add_rmsnorm_mxfp8_row_major(x,r,w,1e-6)
        st=compare_mxfp8(q.cpu().view(torch.uint8),unshuffle_scales(s,rows).cpu(),ref)
        assert st['scale_max_diff']<=1 and st['scale_mismatch']<=SCALE_MISMATCH,st
        assert st['code_far']==0 and st['code_mismatch']<=CODE_MISMATCH,st
        assert torch.equal(res.cpu().view(torch.uint8),rr.view(torch.uint8))
