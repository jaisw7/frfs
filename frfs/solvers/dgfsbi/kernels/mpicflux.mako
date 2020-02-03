# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

<%include file='frfs.solvers.dgfsbi.kernels.rsolvers.${rsolver}'/>

<%frfs:kernel name='mpicflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='in mpi fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'
              cvx='in fpdtype_t'
              cvy='in fpdtype_t'
              cvz='in fpdtype_t'
              varidx='scalar fpdtype_t'>
    // Perform the Riemann solve
    fpdtype_t fn[${nvars}];
    ${frfs.expand('rsolve', 'ul', 'ur', 'nl', 'fn', 'cvx', 'cvy', 'cvz', 'varidx')};

    // Scale and write out the common normal fluxes
% for i in range(nvars):
    ul[${i}] = magnl*fn[${i}];
% endfor
</%frfs:kernel>
