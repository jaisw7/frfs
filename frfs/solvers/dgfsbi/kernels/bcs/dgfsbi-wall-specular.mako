# -*- coding: utf-8 -*-
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

<%frfs:macro name='bc_rsolve_state' params='ul, nl, ur, ploc, cvx, cvy, cvz, 
    bnd_f0, bc_vals, t, varidx'>

// Apply the boundary condition

int vidx = (int) varidx;
int ridx = 0;

% for i in range(nvars):
    ridx = (int) bnd_f0_v[X_IDX + (${i}+vidx)*${c['ninterfpts']}];
    ur[${i}] = bc_vals_v[X_IDX + ridx*${c['ninterfpts']}];
% endfor

</%frfs:macro>
