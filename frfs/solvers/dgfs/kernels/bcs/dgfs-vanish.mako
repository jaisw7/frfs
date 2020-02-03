# -*- coding: utf-8 -*-
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

<%frfs:macro name='bc_rsolve_state' params='ul, nl, ur, ploc, cvx, cvy, cvz, 
    bnd_f0, bc_vals, t, varidx'>

int vidx = (int) varidx;
fpdtype_t fac = 0;
% for i in range(nvars):
    fac = ${' + '.join("(cv{0}_v[{1}+vidx])*(nl[{2}])".format(v, i, j) for j, v in enumerate('xyz'[:ndims]))};

    //ur[${i}] = ul[${i}];
    ur[${i}] = (fac<0)*(0.0) + (fac>=0)*(ul[${i}]);
    //ur[${i}] = (fac<0)*(ul[${i}]) + (fac>=0)*(0.0);
    //ur[${i}] = 0.;
% endfor
</%frfs:macro>
