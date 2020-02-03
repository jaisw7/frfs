# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

<%include file='frfs.solvers.dgfs.kernels.rsolvers.${rsolver}'/>
<%include file='frfs.solvers.dgfs.kernels.bcs.${bctype}'/>

<%frfs:kernel name='bccflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'
              ploc='in fpdtype_t[${str(ndims)}]'
              cvx='in fpdtype_t'
              cvy='in fpdtype_t'
              cvz='in fpdtype_t'
              bnd_f0='in fpdtype_t'
              bc_vals='in fpdtype_t[${str(ndims)}]'
              t='scalar fpdtype_t'
              varidx='scalar fpdtype_t'>
    // Compute the RHS
    fpdtype_t ur[${nvars}];
    ${frfs.expand('bc_rsolve_state', 'ul', 'nl', 'ur', 
    'ploc', 'cvx', 'cvy', 'cvz', 'bnd_f0', 'bc_vals', 't', 'varidx')};

    // Perform the Riemann solve
    fpdtype_t fn[${nvars}];
    ${frfs.expand('rsolve', 'ul', 'ur', 'nl', 'fn', 'cvx', 'cvy', 'cvz', 'varidx')};

    //int vidx = (int) varidx;
    //fpdtype_t nv;

    // Scale and write out the common normal fluxes
% for i in range(nvars):
    ul[${i}] = magnl*fn[${i}];
    
    //nv = ${' + '.join('nl[{j}]*(cv{v}_v[{i}+vidx])'.format(i=i, j=j, v='xyz'[j]) for j in range(ndims))};
    //ul[${i}] = magnl*0.5*(sqrt(nv*nv)+fabs(nv))*(ul[${i}] - ur[${i}]);
% endfor
</%frfs:kernel>
