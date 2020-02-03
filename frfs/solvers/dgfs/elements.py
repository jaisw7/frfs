# -*- coding: utf-8 -*-

from frfs.backends.base.kernels import ComputeMetaKernel
from frfs.solvers.base import BaseElements


class DGFSElements(BaseElements):
    @property
    def _scratch_bufs(self):
        # inside the bufs, one can define new string to allocate on backend
        if 'flux' in self.antialias:
            bufs = {'scal_fpts', 'scal_qpts', 'vect_qpts'}
        elif 'div-flux' in self.antialias:
            bufs = {'scal_fpts', 'vect_upts', 'scal_qpts'}
        else:
            bufs = {'scal_fpts', 'vect_upts'}

        if self._soln_in_src_exprs:
            if 'div-flux' in self.antialias:
                bufs |= {'scal_qpts_cpy'}
            else:
                bufs |= {'scal_upts_cpy'}

        return bufs

    def set_backend(self, backend, nscal_upts, nonce):
        super().set_backend(backend, nscal_upts, nonce)

        # Register pointwise kernels with the backend
        backend.pointwise.register(
            'frfs.solvers.dgfs.kernels.negdivconf'
        )

        # What anti-aliasing options we're running with
        fluxaa = 'flux' in self.antialias
        divfluxaa = 'div-flux' in self.antialias

        # What the source term expressions (if any) are a function of
        plocsrc = self._ploc_in_src_exprs
        solnsrc = self._soln_in_src_exprs

        # Source term kernel arguments
        srctplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'srcex': self._src_exprs
        }

        # Interpolation from elemental points
        if fluxaa or (divfluxaa and solnsrc):
            self.kernels['disu'] = lambda: backend.kernel(
                'mul', self.opmat('M8'), self.scal_upts_inb,
                out=self._scal_fqpts
            )
        else:
            self.kernels['disu'] = lambda: backend.kernel(
                'mul', self.opmat('M0'), self.scal_upts_inb,
                out=self._scal_fpts
            )

        # Interpolations and projections to/from quadrature points
        if divfluxaa:
            self.kernels['tdivf_qpts'] = lambda: backend.kernel(
                'mul', self.opmat('M7'), self.scal_upts_outb,
                out=self._scal_qpts
            )
            self.kernels['divf_upts'] = lambda: backend.kernel(
                'mul', self.opmat('M9'), self._scal_qpts,
                out=self.scal_upts_outb
            )

        # First flux correction kernel
        if fluxaa:
            self.kernels['tdivtpcorf'] = lambda: backend.kernel(
                'mul', self.opmat('(M1 - M3*M2)*M10'), self._vect_qpts,
                out=self.scal_upts_outb
            )
        else:
            self.kernels['tdivtpcorf'] = lambda: backend.kernel(
                'mul', self.opmat('M1 - M3*M2'), self._vect_upts,
                out=self.scal_upts_outb
            )

        # Second flux correction kernel
        self.kernels['tdivtconf'] = lambda: backend.kernel(
            'mul', self.opmat('M3'), self._scal_fpts, out=self.scal_upts_outb,
            beta=1.0
        )

        # Transformed to physical divergence kernel + source term
        if divfluxaa:
            plocqpts = self.ploc_at('qpts') if plocsrc else None
            solnqpts = self._scal_qpts_cpy if solnsrc else None

            if solnsrc:
                self.kernels['copy_soln'] = lambda: backend.kernel(
                    'copy', self._scal_qpts_cpy, self._scal_qpts
                )

            self.kernels['negdivconf'] = lambda: backend.kernel(
                'negdivconf', tplargs=srctplargs,
                dims=[self.nqpts, self.neles], tdivtconf=self._scal_qpts,
                rcpdjac=self.rcpdjac_at('qpts'), ploc=plocqpts, u=solnqpts
            )
        else:
            plocupts = self.ploc_at('upts') if plocsrc else None
            solnupts = self._scal_upts_cpy if solnsrc else None

            if solnsrc:
                self.kernels['copy_soln'] = lambda: backend.kernel(
                    'copy', self._scal_upts_cpy, self.scal_upts_inb
                )

            self.kernels['negdivconf'] = lambda: backend.kernel(
                'negdivconf', tplargs=srctplargs,
                dims=[self.nupts, self.neles], tdivtconf=self.scal_upts_outb,
                rcpdjac=self.rcpdjac_at('upts'), ploc=plocupts, u=solnupts
            )

        # In-place solution filter
        if self.cfg.getint('soln-filter', 'nsteps', '0'):
            def filter_soln():
                mul = backend.kernel(
                    'mul', self.opmat('M11'), self.scal_upts_inb,
                    out=self._scal_upts_temp
                )
                copy = backend.kernel(
                    'copy', self.scal_upts_inb, self._scal_upts_temp
                )

                return ComputeMetaKernel([mul, copy])

            self.kernels['filter_soln'] = filter_soln

        # Added for dgfs
        # Register our flux kernel
        backend.pointwise.register('frfs.solvers.dgfs.kernels.tflux')

        # Template parameters for the flux kernel
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, 
                        vsize=self._vm.vsize(),
                       c=self.cfg.items_as('constants', float))

        if 'flux' in self.antialias:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nqpts, self.neles],
                u=self._scal_qpts, smats=self.smat_at('qpts'),
                f=self._vect_qpts, 
                cvx=self._vm.d_cvx(), cvy=self._vm.d_cvy(), 
                cvz=self._vm.d_cvz()
            )
        else:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nupts, self.neles],
                u=self.scal_upts_inb, smats=self.smat_at('upts'),
                f=self._vect_upts,
                cvx=self._vm.d_cvx(), cvy=self._vm.d_cvy(), 
                cvz=self._vm.d_cvz()
            )

        self.kernels['invmass'] = lambda:backend.kernel(
            'mul', self.opmat('M12'), self.scal_upts_inb,
            out=self.scal_upts_outb
        )


    # allow standard single-step integrators
    formulations = ['std']

    #distvars = ['f_'+str(i) for i in range(2)];
    #privarmap = {1: distvars, 2: distvars, 3: distvars}
    #distvarsvis=[(ivar,[ivar]) for ivar in distvars]
    #visvarmap = {1: distvarsvis, 2: distvarsvis, 3: distvarsvis}
    #convarmap = {1: distvars, 2: distvars, 3: distvars}

    privarmap = {
        1: ['rho', 'U:x', 'U:y', 'T', 'Q:x', 'p'],
        2: ['rho', 'U:x', 'U:y', 'T', 'Q:x', 'Q:y', 
            'P:xx', 'P:xy', 'P:yy', 'p'],
        3: ['rho', 'U:x', 'U:y', 'U:z', 'T', 'Q:x', 'Q:y', 'Q:z', 
            'P:xx', 'P:xy', 'P:xz', 'P:yy', 'P:yz', 'P:zz', 'p']
    }
    convarmap = privarmap
    dualcoeffs = convarmap
    visvarmap = {
        1: [('density', ['rho']),
            ('velocity', ['U:x', 'U:y']),
            ('temperature', ['T']),
            ('heat-flux', ['Q:x']),
            ('pressure', ['p'])],
        2: [('density', ['rho']),
            ('velocity', ['U:x', 'U:y']),
            ('temperature', ['T']),
            ('heat-flux', ['Q:x', 'Q:y']), 
            ('pressure-tensor', ['P:xx', 'P:xy', 'P:yy']),
            ('pressure', ['p'])],
        3: [('density', ['rho']),
            ('velocity', ['U:x', 'U:y', 'U:z']),
            ('temperature', ['T']),
            ('heat-flux', ['Q:x', 'Q:y', 'Q:z']), 
            ('pressure-tensor', ['P:xx','P:xy','P:xz','P:yy','P:yz','P:zz']),
            ('pressure', ['p'])]
    }

    def __init__(self, basiscls, eles, cfg, vm=None):
        self._vm = vm

        # must update DGFSInflowBCInters if distvars defn here is updated
        # must update dgfsdistwriter if distvars defn here is updated
        distvars = ['f_'+str(i) 
            for i in range(self._vm.NvBatchSize())];

        # variables in 1D, 2D, 3D
        self.privarmap = {1: distvars, 2: distvars, 3: distvars}
        self.convarmap = {1: distvars, 2: distvars, 3: distvars}
        self.dualcoeffs = self.convarmap

        super().__init__(basiscls, eles, cfg)


    @staticmethod
    def pri_to_con(pris, cfg):
        return pris
        #return [pris[0]*cfg.getfloat('constants', 'advx')]

    @staticmethod
    def con_to_pri(cons, cfg):
        return cons
        #return cons[0]/cfg.getfloat('constants', 'advx')
