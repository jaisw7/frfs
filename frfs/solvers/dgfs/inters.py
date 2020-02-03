# -*- coding: utf-8 -*-

import math

from frfs.solvers.base import BaseInters, get_opt_view_perm
from frfs.nputil import npeval
import numpy as np
from frfs.solvers.dgfs.initcond import DGFSInitCondition
from frfs.util import subclass_where

# need to fix this (to make things backend independent)
from pycuda import compiler, gpuarray
from frfs.template import DottedTemplateLookup
import pycuda.driver as cuda
from frfs.backends.cuda.provider import get_grid_for_block

class DGFSIntInters(BaseInters):
    def __init__(self, be, lhs, rhs, elemap, cfg, vm=None):
        super().__init__(be, lhs, elemap, cfg)

        if(vm==None):
            raise ValueError("Needs velocity mesh")
        self._vm = vm

        const_mat = self._const_mat

        # Compute the `optimal' permutation for our interface
        self._gen_perm(lhs, rhs)

        # Generate the left and right hand side view matrices
        self._scal_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')
        self._scal_rhs = self._scal_view(rhs, 'get_scal_fpts_for_inter')

        # Generate the constant matrices
        self._mag_pnorm_lhs = const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._norm_pnorm_lhs = const_mat(lhs, 'get_norm_pnorms_for_inter')

        # Added for dgfs
        self._be.pointwise.register('frfs.solvers.dgfs.kernels.intcflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c, vsize=self._vm.vsize())

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs, 
            cvx=self._vm.d_cvx(), 
            cvy=self._vm.d_cvy(), 
            cvz=self._vm.d_cvz()
        )

    def _gen_perm(self, lhs, rhs):
        # Arbitrarily, take the permutation which results in an optimal
        # memory access pattern for the LHS of the interface
        self._perm = get_opt_view_perm(lhs, 'get_scal_fpts_for_inter',
                                       self._elemap)


class DGFSMPIInters(BaseInters):
    # Tag used for MPI
    MPI_TAG = 2314

    def __init__(self, be, lhs, rhsrank, rallocs, elemap, cfg, vm=None):
        super().__init__(be, lhs, elemap, cfg)

        if(vm==None):
            raise ValueError("Need velocity mesh")
        self._vm = vm

        self._rhsrank = rhsrank
        self._rallocs = rallocs

        const_mat = self._const_mat

        # Generate the left hand view matrix and its dual
        self._scal_lhs = self._scal_xchg_view(lhs, 'get_scal_fpts_for_inter')
        self._scal_rhs = be.xchg_matrix_for_view(self._scal_lhs)

        self._mag_pnorm_lhs = const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._norm_pnorm_lhs = const_mat(lhs, 'get_norm_pnorms_for_inter')

        # Kernels
        self.kernels['scal_fpts_pack'] = lambda: be.kernel(
            'pack', self._scal_lhs
        )
        self.kernels['scal_fpts_send'] = lambda: be.kernel(
            'send_pack', self._scal_lhs, self._rhsrank, self.MPI_TAG
        )
        self.kernels['scal_fpts_recv'] = lambda: be.kernel(
            'recv_pack', self._scal_rhs, self._rhsrank, self.MPI_TAG
        )
        self.kernels['scal_fpts_unpack'] = lambda: be.kernel(
            'unpack', self._scal_rhs
        )

        # Added for dgfs
        self._be.pointwise.register('frfs.solvers.dgfs.kernels.mpicflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c, vsize=self._vm.vsize())

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            cvx=self._vm.d_cvx(),
            cvy=self._vm.d_cvy(),
            cvz=self._vm.d_cvz()
        )


class DGFSBCInters(BaseInters):
    type = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg, vm=None):
        super().__init__(be, lhs, elemap, cfg)

        if(vm==None):
            raise ValueError("Need velocity mesh")
        self._vm = vm

        self.cfgsect = cfgsect

        const_mat = self._const_mat

        # For BC interfaces, which only have an LHS state, we take the
        # permutation which results in an optimal memory access pattern
        # iterating over this state.
        self._perm = get_opt_view_perm(lhs, 'get_scal_fpts_for_inter', elemap)

        # LHS view and constant matrices
        self._scal_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')
        self._mag_pnorm_lhs = const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._norm_pnorm_lhs = const_mat(lhs, 'get_norm_pnorms_for_inter')
        #self._ploc = None
        self._ploc = self._const_mat(lhs, 'get_ploc_for_inter')
        
    def _eval_opts(self, opts, default=None):
        # Boundary conditions, much like initial conditions, can be
        # parameterized by values in [constants] so we must bring these
        # into scope when evaluating the boundary conditions
        cc = self.cfg.items_as('constants', float)

        cfg, sect = self.cfg, self.cfgsect

        # Evaluate any BC specific arguments from the config file
        if default is not None:
            return [npeval(cfg.getexpr(sect, k, default), cc) for k in opts]
        else:
            return [npeval(cfg.getexpr(sect, k), cc) for k in opts]

    def _exp_opts(self, opts, lhs, default={}):
        cfg, sect = self.cfg, self.cfgsect

        subs = cfg.items('constants')
        subs.update(x='ploc[0]', y='ploc[1]', z='ploc[2]')
        subs.update(abs='fabs', pi=str(math.pi))

        exprs = {}
        for k in opts:
            if k in default:
                exprs[k] = cfg.getexpr(sect, k, default[k], subs=subs)
            else:
                exprs[k] = cfg.getexpr(sect, k, subs=subs)

        if any('ploc' in ex for ex in exprs.values()) and not self._ploc:
            self._ploc = self._const_mat(lhs, 'get_ploc_for_inter')

        return exprs


class DGFSWallDiffuseBCInters(DGFSBCInters):
    type = 'dgfs-wall-diffuse'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, **kwargs):
        
        super().__init__(be, lhs, elemap, cfgsect, cfg, **kwargs)

        initcondcls = subclass_where(DGFSInitCondition, model='maxwellian')
        bc = initcondcls(be, cfg, self._vm, cfgsect)
        f0 = bc.get_init_vals().reshape(1, self._vm.vsize())
        self._d_bnd_f0 = be.const_matrix(f0)
        unondim = bc.unondim()

        tplc = {
            'Ux': unondim[0,0],
            'Uy': unondim[1,0],
            'Uz': unondim[2,0]
        }
        self._tpl_c.update(tplc)

        # storage
        #self._bcVals0_lhs = self._const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._bcVals_lhs = self._mat(lhs, 'get_norm_pnorms_for_inter', 1.)
        #self._bcVals_lhs = self._scal_view(lhs, 'get_norm_pnorms_for_inter')
        #self._bcVals_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')
        #self._bcVals_lhs = self._view(lhs, 
        #    'get_mag_pnorms_for_inter', vshape=(1,))        

        # register comm_flux kernel
        self._be.pointwise.register('frfs.solvers.dgfs.kernels.bccflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                        c=self._tpl_c, bctype=self.type, 
                        vsize=self._vm.vsize(), 
                        bcupdatetype=self.type+'-update', cw=self._vm.cw()
                    )

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs, dims=[self.ninterfpts], ul=self._scal_lhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            ploc=self._ploc, 
            cvx=self._vm.d_cvx(), cvy=self._vm.d_cvy(), cvz=self._vm.d_cvz(), 
            bnd_f0=self._d_bnd_f0, bc_vals=self._bcVals_lhs
        )

        # register update for boundary conditions
        self._be.pointwise.register('frfs.solvers.dgfs.kernels.bccupdate')

        self.kernels['update_bc'] = lambda: self._be.kernel(
            'bccupdate', tplargs, dims=[self.ninterfpts], ul=self._scal_lhs,
            nl=self._norm_pnorm_lhs, ploc=self._ploc, 
            cvx=self._vm.d_cvx(), cvy=self._vm.d_cvy(), cvz=self._vm.d_cvz(),
            bnd_f0=self._d_bnd_f0, bc_vals=self._bcVals_lhs
        )


class DGFSWallSpecularBCInters(DGFSBCInters):
    type = 'dgfs-wall-specular'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, **kwargs):
        
        super().__init__(be, lhs, elemap, cfgsect, cfg, **kwargs)

        norms = self._norm_pnorm_lhs.get()
        isCurved = False
        # check if the norms are same (non-curved surface)
        for cmpt in range(norms.shape[0]):
            #print(norms[cmpt,:])
            #print(norms[cmpt,0])
            #print(norms[cmpt,:]==norms[cmpt,0])
            #assert np.allclose(norms[cmpt,:], norms[cmpt,0]), (
            #    "Need non-curved boundaries, rotate line loops!")
            if not np.allclose(norms[cmpt,:], norms[cmpt,0]):
                isCurved = True

        
        tplc = {'ninterfpts': self.ninterfpts}
        self._tpl_c.update(tplc)

        ndims=self.ndims
        if ndims==2:
            norms = np.vstack((norms, np.zeros((1, self.ninterfpts))))
        ndims=3
        vsize = self._vm.vsize()
        cv = self._vm.cv()
        f0 = np.zeros((vsize,self.ninterfpts))
        if isCurved:
            for fpts in range(self.ninterfpts):
                norm = norms[0:ndims,fpts]
                for j in range(vsize):
                    dot = np.dot(norm, cv[0:ndims,j])
                    cr = cv[0:ndims,j] - 2*dot*norm
                    diff = cr.reshape(ndims,1)-cv[0:ndims,:]
                    f0[j,fpts] = np.argmin(np.sum(np.abs(diff), axis=0))
        else:
            norm = norms[0:ndims,0]
            for j in range(vsize):
                dot = np.dot(norm, cv[0:ndims,j])
                cr = cv[0:ndims,j] - 2*dot*norm
                diff = cr.reshape(ndims,1)-cv[0:ndims,:]
                #f0[j,0] = np.argmin(np.sum(np.abs(diff), axis=0))
                f0[j,0] = np.argmin(np.sum(diff**2, axis=0))
            #f0 = np.tile(f0[:,0], (self.ninterfpts, 1))
            for fpts in range(self.ninterfpts):
                f0[:, fpts] = f0[:, 0] 

        # some sanity checks (unique reverse map)
        #sorted_f0 = np.sort(f0[:,0])
        #print(np.intersect1d(sorted_f0, np.arange(0,vsize)).shape)
        assert np.all(np.sort(f0[:,0])==np.arange(0,vsize)), "Non-unique map"
        #raise ValueError("")

        f0 = f0.reshape(vsize*self.ninterfpts, 1)
        self._d_bnd_f0 = be.const_matrix(f0)

        # storage
        self._bcVals_lhs = be.matrix(f0.shape, f0)

        # register comm_flux kernel
        self._be.pointwise.register('frfs.solvers.dgfs.kernels.bccflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c, bctype=self.type, 
                       vsize=self._vm.vsize(), cw=self._vm.cw(),
                       bcupdatetype=self.type+'-update'
                    )

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs, dims=[self.ninterfpts], ul=self._scal_lhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            ploc=self._ploc, 
            cvx=self._vm.d_cvx(), cvy=self._vm.d_cvy(), cvz=self._vm.d_cvz(), 
            bnd_f0=self._d_bnd_f0, bc_vals=self._bcVals_lhs
        )

        # register update for boundary conditions
        self._be.pointwise.register('frfs.solvers.dgfs.kernels.bccupdate')

        self.kernels['update_bc'] = lambda: self._be.kernel(
            'bccupdate', tplargs, dims=[self.ninterfpts], ul=self._scal_lhs,
            nl=self._norm_pnorm_lhs, ploc=self._ploc, 
            cvx=self._vm.d_cvx(), cvy=self._vm.d_cvy(), cvz=self._vm.d_cvz(),
            bnd_f0=self._d_bnd_f0, bc_vals=self._bcVals_lhs
        )


class DGFSInletBCInters(DGFSBCInters):
    type = 'dgfs-inlet'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, **kwargs):
        
        super().__init__(be, lhs, elemap, cfgsect, cfg, **kwargs)

        initcondcls = subclass_where(DGFSInitCondition, model='maxwellian')
        bc = initcondcls(be, cfg, self._vm, cfgsect)
        f0 = bc.get_init_vals().reshape(1, self._vm.vsize())
        self._d_bnd_f0 = be.const_matrix(f0)
        unondim = bc.unondim()

        tplc = {
            'Ux': unondim[0,0],
            'Uy': unondim[1,0],
            'Uz': unondim[2,0]
        }
        self._tpl_c.update(tplc)

        # storage
        self._bcVals_lhs = None

        # register comm_flux kernel
        self._be.pointwise.register('frfs.solvers.dgfs.kernels.bccflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c, bctype=self.type, 
                       vsize=self._vm.vsize(), cw=self._vm.cw()
                    )

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs, dims=[self.ninterfpts], ul=self._scal_lhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            ploc=self._ploc, 
            cvx=self._vm.d_cvx(), cvy=self._vm.d_cvy(), cvz=self._vm.d_cvz(), 
            bnd_f0=self._d_bnd_f0, bc_vals=self._bcVals_lhs
        )


class DGFSInletNormalShockBCInters(DGFSBCInters):
    type = 'dgfs-inlet-normalshock'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, **kwargs):
        
        super().__init__(be, lhs, elemap, cfgsect, cfg, **kwargs)

        initcondcls = subclass_where(DGFSInitCondition, model='maxwellian')
        bc = initcondcls(be, cfg, self._vm, cfgsect)
        f0 = bc.get_init_vals().reshape(1, self._vm.vsize())
        self._d_bnd_f0 = be.const_matrix(f0)

        # storage
        self._bcVals_lhs = None

        # register comm_flux kernel
        self._be.pointwise.register('frfs.solvers.dgfs.kernels.bccflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c, bctype=self.type, 
                       vsize=self._vm.vsize(), cw=self._vm.cw()
                    )

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs, dims=[self.ninterfpts], ul=self._scal_lhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            ploc=self._ploc, 
            cvx=self._vm.d_cvx(), cvy=self._vm.d_cvy(), cvz=self._vm.d_cvz(), 
            bnd_f0=self._d_bnd_f0, bc_vals=self._bcVals_lhs
        )


class DGFSOutletBCInters(DGFSBCInters):
    type = 'dgfs-outlet'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, **kwargs):
        
        super().__init__(be, lhs, elemap, cfgsect, cfg, **kwargs)

        self._d_bnd_f0 = None

        # storage
        self._bcVals_lhs = None

        # register comm_flux kernel
        self._be.pointwise.register('frfs.solvers.dgfs.kernels.bccflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c, bctype=self.type, 
                       vsize=self._vm.vsize(), cw=self._vm.cw()
                    )

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs, dims=[self.ninterfpts], ul=self._scal_lhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            ploc=self._ploc, 
            cvx=self._vm.d_cvx(), cvy=self._vm.d_cvy(), cvz=self._vm.d_cvz(), 
            bnd_f0=self._d_bnd_f0, bc_vals=self._bcVals_lhs
        )


class DGFSVanishBCInters(DGFSBCInters):
    type = 'dgfs-vanish'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, **kwargs):
        
        super().__init__(be, lhs, elemap, cfgsect, cfg, **kwargs)

        self._d_bnd_f0 = None

        # storage
        self._bcVals_lhs = None

        # register comm_flux kernel
        self._be.pointwise.register('frfs.solvers.dgfs.kernels.bccflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c, bctype=self.type, 
                       vsize=self._vm.vsize(), cw=self._vm.cw()
                    )

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs, dims=[self.ninterfpts], ul=self._scal_lhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            ploc=self._ploc, 
            cvx=self._vm.d_cvx(), cvy=self._vm.d_cvy(), cvz=self._vm.d_cvz(), 
            bnd_f0=self._d_bnd_f0, bc_vals=self._bcVals_lhs
        )


class DGFSWallDiffuseCylBCInters(DGFSBCInters):
    type = 'dgfs-wall-diffuse-cyl'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, **kwargs):
        
        super().__init__(be, lhs, elemap, cfgsect, cfg, **kwargs)

        tplc = self._exp_opts(
            ['ux', 'uy', 'uz', 'T'], lhs
        )
        tplc['Ux'] = '((' + tplc['ux'] + ')/' + str(self._vm.u0()) + ')'
        tplc['Uy'] = '((' + tplc['uy'] + ')/' + str(self._vm.u0()) + ')'
        tplc['Uz'] = '((' + tplc['uz'] + ')/' + str(self._vm.u0()) + ')'
        tplc['T'] = '((' + tplc['T'] + ')/' + str(self._vm.T0()) + ')'
        tplc.pop('ux'); tplc.pop('uy'); tplc.pop('uz');
        self._tpl_c.update(tplc)

        # storage
        self._bcVals_lhs = self._mat(lhs, 'get_norm_pnorms_for_inter', 1.)

        # register comm_flux kernel
        self._be.pointwise.register('frfs.solvers.dgfs.kernels.bccflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                        c=self._tpl_c, bctype=self.type, 
                        vsize=self._vm.vsize(), 
                        bcupdatetype=self.type+'-update', cw=self._vm.cw()
                    )

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs, dims=[self.ninterfpts], ul=self._scal_lhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            ploc=self._ploc, 
            cvx=self._vm.d_cvx(), cvy=self._vm.d_cvy(), cvz=self._vm.d_cvz(), 
            bnd_f0=self._vm.d_cvz(), bc_vals=self._bcVals_lhs
        )

        # register update for boundary conditions
        self._be.pointwise.register('frfs.solvers.dgfs.kernels.bccupdate')

        self.kernels['update_bc'] = lambda: self._be.kernel(
            'bccupdate', tplargs, dims=[self.ninterfpts], ul=self._scal_lhs,
            nl=self._norm_pnorm_lhs, ploc=self._ploc, 
            cvx=self._vm.d_cvx(), cvy=self._vm.d_cvy(), cvz=self._vm.d_cvz(),
            bnd_f0=self._vm.d_cvz(), bc_vals=self._bcVals_lhs
        )
