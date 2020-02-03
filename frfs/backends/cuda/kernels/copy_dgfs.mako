# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

__global__ void
copy_dgfs(int nrow, int ncolb, int ldim0, int ldim1, 
       fpdtype_t* __restrict__ x0,
       fpdtype_t* __restrict__ x1,
       int sidx0, int eidx0, 
       int sidx1, int eidx1
)
{
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int idx0, idx1;

    if (j < ncolb && i < nrow)
    {
    % for k in subdims:
        idx0 = i*ldim0 + SOA_IX(j, ${k}+sidx0, ${ncola0});
        idx1 = i*ldim1 + SOA_IX(j, ${k}+sidx1, ${ncola1});

        x1[idx1] = x0[idx0];
    % endfor
    }
}
