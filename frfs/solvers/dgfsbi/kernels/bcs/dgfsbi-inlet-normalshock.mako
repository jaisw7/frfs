# -*- coding: utf-8 -*-
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

<%frfs:macro name='bc_rsolve_state' params='ul, nl, ur, ploc, cvx, cvy, cvz, 
    bnd_f0, bc_vals, t, varidx'>

    // Apply the boundary condition

int vidx = (int) varidx;
fpdtype_t fac = 0;

% for i in range(nvars):
    //ur[${i}] = bnd_f0[${i} + vidx];
    fac = ${' + '.join("(cv{0}_v[{1}+vidx])*(nl[{2}])".format(v, i, j) for j, v in enumerate('xyz'[:ndims]))};
    //ur[${i}] = (fac<0)*(bnd_f0_v[${i}+vidx]) + (fac>=0)*(0.0);
    ur[${i}] = (fac<0)*(bnd_f0_v[${i}+vidx]) + (fac>=0)*(ul[${i}]);
% endfor
</%frfs:macro>
