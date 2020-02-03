# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

<%include file='frfs.solvers.dgfsbi.kernels.rsolvers.${rsolver}'/>
<%include file='frfs.solvers.dgfsbi.kernels.bcs.${bcupdatetype}'/>

<%frfs:kernel name='bccupdate' ndim='1'
              ul='in view fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              cvx='in fpdtype_t'
              cvy='in fpdtype_t'
              cvz='in fpdtype_t'
              bnd_f0='in fpdtype_t'
              bc_vals='out fpdtype_t[${str(ndims)}]'
              t='scalar fpdtype_t'
              varidx='scalar fpdtype_t'>
    // Compute the RHS
    ${frfs.expand('bc_rsolve_state_update', 'ul', 'nl', 
      'ploc', 'cvx', 'cvy', 'cvz', 'bnd_f0', 'bc_vals', 
      't', 'varidx')};

</%frfs:kernel>
