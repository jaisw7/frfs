# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

__global__ void
axnpby(int nrow, int ncolb, int ldim, fpdtype_t* __restrict__ x0,
       ${', '.join('const fpdtype_t* __restrict__ x' + str(i)
                   for i in range(1, nv))},
       ${', '.join('fpdtype_t a' + str(i) for i in range(nv))})
{
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int idx;

    if (j < ncolb && a0 == 0.0)
    {
    % for k in subdims:
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] = ${frfs.dot('a{l}', 'x{l}[idx]', l=(1, nv))};
    % endfor
    }
    else if (j < ncolb && a0 == 1.0)
    {
    % for k in subdims:
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] += ${frfs.dot('a{l}', 'x{l}[idx]', l=(1, nv))};
    % endfor
    }
    else if (j < ncolb)
    {
    % for k in subdims:
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] = ${frfs.dot('a{l}', 'x{l}[idx]', l=nv)};
    % endfor
    }
}
