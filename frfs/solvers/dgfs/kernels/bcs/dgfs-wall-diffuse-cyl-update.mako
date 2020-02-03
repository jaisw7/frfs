# -*- coding: utf-8 -*-
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

<%frfs:macro name='bc_rsolve_state_update' params='ul, nl, ploc, 
    cvx, cvy, cvz, bnd_f0, bc_vals, t, 
    varidx'>

    // Apply the boundary condition

int vidx = (int) varidx;

fpdtype_t fac = 0, ur_num = 0, ur_den = 0, mxwl = 0;

// All the threads will execute only one of the two blocks
if(vidx==0)
{
    bc_vals[0] = 0; bc_vals[1] = 0;
}

% for i in range(nvars):

    fac = ${
        ' + '.join("(cv{0}_v[{1}+vidx]-({3}))*(nl[{2}])".format(v, i, j, c['U'+v])
            for j, v in enumerate('xyz'[:ndims]))
    };

    mxwl = exp(-(
                (cvx_v[${i}+vidx]-${c['Ux']})*(cvx_v[${i}+vidx]-${c['Ux']})
                +(cvy_v[${i}+vidx]-${c['Uy']})*(cvy_v[${i}+vidx]-${c['Uy']})
                +(bnd_f0_v[${i}+vidx]-${c['Uz']})*(bnd_f0_v[${i}+vidx]-${c['Uz']})
           )/${c['T']});

    ur_num += (fac>=0)*(fac*ul[${i}]*${cw});
    ur_den += (fac<0)*(fac*mxwl*${cw});
    
    //ur_num += (fac>=0)*1;
    //ur_den += (fac<0)*1;

% endfor

// update
bc_vals[0] += ur_num; bc_vals[1] += ur_den;
//bc_vals[0] = ${c['Ux']}; bc_vals[1] = ${c['Uy']};
//bc_vals[0] = nl[0]; bc_vals[1] = nl[1];

</%frfs:macro>
