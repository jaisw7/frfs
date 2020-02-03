# -*- coding: utf-8 -*-
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

<%frfs:macro name='bc_rsolve_state' params='ul, nl, ur, ploc, cvx, cvy, cvz, 
    bnd_f0, bc_vals, t, varidx'>

    // Apply the boundary condition

int vidx = (int) varidx;
fpdtype_t fac = 0, mxwl = 0;


% for i in range(nvars):
    fac = ${' + '.join("(cv{0}_v[{1}+vidx]-({3}))*(nl[{2}])".format(v, i, j, c['U'+v]) for j, v in enumerate('xyz'[:ndims]))};
    //ur[${i}] = (fac<0)*(bnd_f0_v[${i}+vidx]*(-bc_vals[0]/bc_vals[1])) + (fac>=0)*(0.0);
    
    //~ sorta works, the density seems to be increasing
    mxwl = exp(-(
                (cvx_v[${i}+vidx]-${c['Ux']})*(cvx_v[${i}+vidx]-${c['Ux']})
                +(cvy_v[${i}+vidx]-${c['Uy']})*(cvy_v[${i}+vidx]-${c['Uy']})
                +(bnd_f0_v[${i}+vidx]-${c['Uz']})*(bnd_f0_v[${i}+vidx]-${c['Uz']})
           )*${mr}/${c['T']});

    ur[${i}] = (fac<0)*(mxwl*(-bc_vals[0]/bc_vals[1])) + (fac>=0)*(ul[${i}]);

    //ur[${i}] = bnd_f0_v[${i}+vidx]*(-bc_vals[0]/bc_vals[1]);
% endfor

</%frfs:macro>
