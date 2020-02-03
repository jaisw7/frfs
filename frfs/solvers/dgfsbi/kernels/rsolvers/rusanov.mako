# -*- coding: utf-8 -*-
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>
<%include file='frfs.solvers.dgfsbi.kernels.flux'/>

<%frfs:macro name='rsolve' params='ul, ur, n, nf, cvx, cvy, cvz, varidx'>
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];

    ${frfs.expand('inviscid_flux', 'ul', 'fl', 'cvx', 'cvy', 'cvz', 'varidx')};
    ${frfs.expand('inviscid_flux', 'ur', 'fr', 'cvx', 'cvy', 'cvz', 'varidx')};

    //fpdtype_t vl[${ndims}] = ${frfs.array('ul[{i}]', i=(0, ndims))};
    //fpdtype_t vr[${ndims}] = ${frfs.array('ur[{i}]', i=(0, ndims))};
    //fpdtype_t nv = 0.5*${frfs.dot('n[{i}]', 'vl[{i}] + vr[{i}]', i=ndims)};

    // Estimate the wave speed (fairly reasonable)
    //fpdtype_t a = fabs(nv);
    //fpdtype_t a = fabs(nv) + sqrt(nv*nv);
    //fpdtype_t a = fabs(nv) + sqrt(nv*nv + cvx_v[0]*cvx_v[0] 
    //    + cvy_v[0]*cvy_v[0] + cvz_v[0]*cvz_v[0]);
    //fpdtype_t a = fabs(nv) + sqrt(nv*nv + cvx_v[0]*cvx_v[0]);
    //fpdtype_t a = fabs(nv) + sqrt(cvx_v[0]*cvx_v[0]);

    int vidx = (int) varidx;
    fpdtype_t a;
    //fpdtype_t nv;
    //fpdtype_t a = sqrt(3*cvx_v[0]*cvx_v[0]);

    // Output
% for i in range(nvars):

    //a = fabs(nv) + sqrt(nv*nv + cvx_v[${i}+vidx]*cvx_v[${i}+vidx] + cvy_v[${i}+vidx]*cvy_v[${i}+vidx]+cvz_v[${i}+vidx]*cvz_v[${i}+vidx]);
    //a = sqrt(cvx_v[${i}+vidx]*cvx_v[${i}+vidx] + cvy_v[${i}+vidx]*cvy_v[${i}+vidx]+cvz_v[${i}+vidx]*cvz_v[${i}+vidx]);
    //nv = ${' + '.join('n[{j}]*(cv{v}_v[{i}+vidx])'.format(i=i, j=j, v='xyz'[j]) for j in range(ndims))};
    //a = fabs(nv) + sqrt(nv*nv);

    // More reasonable estimate
    //a = sqrt(cvx_v[${i}+vidx]*cvx_v[${i}+vidx] + cvy_v[${i}+vidx]*cvy_v[${i}+vidx]+cvz_v[${i}+vidx]*cvz_v[${i}+vidx]);

    a = fabs(${' + '.join('n[{j}]*(cv{v}_v[{i}+vidx])'.format(i=i, j=j, v='xyz'[j]) for j in range(ndims))});
    nf[${i}] = 0.5*(${' + '.join('n[{j}]*(fl[{j}][{i}] + fr[{j}][{i}])'.format(i=i, j=j) for j in range(ndims))})
             + 0.5*a*(ul[${i}] - ur[${i}]);

    //nf[${i}] = 0.5*(${' + '.join('n[{j}]*(cv{v}_v[{i}+vidx])'.format(i=i, j=j, v='xyz'[j]) for j in range(ndims))})*(ul[${i}] - ur[${i}]);
% endfor
</%frfs:macro>
