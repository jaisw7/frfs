# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>
<%include file='frfs.solvers.dgfs.kernels.flux'/>

<%frfs:kernel name='tflux' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              f='out fpdtype_t[${str(ndims)}][${str(nvars)}]'
              cvx='in fpdtype_t'
              cvy='in fpdtype_t'
              cvz='in fpdtype_t'
              varidx='scalar fpdtype_t'>

    // Compute the flux
    fpdtype_t ftemp[${ndims}][${nvars}];
    ${frfs.expand('inviscid_flux', 'u', 'ftemp', 'cvx', 'cvy', 'cvz', 'varidx')};

    // Transform the fluxes
% for i, j in frfs.ndrange(ndims, nvars):
    f[${i}][${j}] = ${' + '.join('smats[{0}][{1}]*ftemp[{1}][{2}]'
                                 .format(i, k, j)
                                 for k in range(ndims))};
% endfor
</%frfs:kernel>
