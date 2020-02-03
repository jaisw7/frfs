# -*- coding: utf-8 -*-
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

<%frfs:macro name='inviscid_flux' params='s, f, cvx, cvy, cvz, varidx'>

//fpdtype_t adv[${ndims}];
// for i in range(ndims):
//    adv[{i}] = {c['adv'+str(i)]};
// endfor

int vidx = (int) varidx;

// Advection flux
% for i, j in frfs.ndrange(ndims, nvars):
    f[${i}][${j}] = ${'cv{0}_v[{1}+vidx]*s[{1}]'.format('xyz'[i],j)};
% endfor

//{c['cv_'+str(i)+str(j+varidx)]}

</%frfs:macro>
