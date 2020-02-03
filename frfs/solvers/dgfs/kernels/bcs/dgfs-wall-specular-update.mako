# -*- coding: utf-8 -*-
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

<%frfs:macro name='bc_rsolve_state_update' params='ul, nl, ploc, 
    cvx, cvy, cvz, bnd_f0, bc_vals, t, 
    varidx'>

    // Apply the boundary condition

int vidx = (int) varidx;

% for i in range(nvars):
    bc_vals_v[X_IDX + (${i}+vidx)*${c['ninterfpts']}] = ul[${i}];
% endfor

</%frfs:macro>
