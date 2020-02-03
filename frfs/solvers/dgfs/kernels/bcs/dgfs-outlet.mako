# -*- coding: utf-8 -*-
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

<%frfs:macro name='bc_rsolve_state' params='ul, nl, ur, ploc, cvx, cvy, cvz, 
    bnd_f0, bc_vals, t, varidx'>
% for i in range(nvars):
    ur[${i}] = ul[${i}];
% endfor
</%frfs:macro>
