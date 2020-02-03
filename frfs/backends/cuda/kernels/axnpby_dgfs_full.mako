# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

__global__ void
axnpby_dgfs_full(int size, fpdtype_t* __restrict__ x0,
       ${', '.join('const fpdtype_t* __restrict__ x' + str(i)
                   for i in range(1, nv))},
       ${', '.join('fpdtype_t a' + str(i) for i in range(nv))})
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < size && a0 == 0.0)
    {
        x0[idx] = ${frfs.dot('a{l}', 'x{l}[idx]', l=(1, nv))};
    }
    else if (idx < size && a0 == 1.0)
    {
        x0[idx] += ${frfs.dot('a{l}', 'x{l}[idx]', l=(1, nv))};
    }
    else if (idx < size)
    {
        x0[idx] = ${frfs.dot('a{l}', 'x{l}[idx]', l=nv)};
    }
}
