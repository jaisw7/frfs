from abc import ABCMeta, abstractmethod
import numpy as np
from math import gamma, isnan, ceil

# need to fix this (to make things backend independent)
from pycuda import compiler, gpuarray
from frfs.template import DottedTemplateLookup
from frfs.solvers.dgfs.cufft import (cufftPlan3d, cufftPlanMany, 
                            cufftExecD2Z, cufftExecZ2Z,
                            CUFFT_D2Z, CUFFT_Z2Z, CUFFT_FORWARD, CUFFT_INVERSE
                        )
import pycuda.driver as cuda
from frfs.backends.cuda.provider import get_grid_for_block
from frfs.solvers.dgfs.cublas import CUDACUBLASKernels
from frfs.util import ndrange

def get_kernel(module, funcname, args):
    func = module.get_function(funcname)
    func.prepare(args)
    func.set_cache_config(cuda.func_cache.PREFER_L1)
    return func

class ADGFSScatteringModel(object, metaclass=ABCMeta):
    def __init__(self, backend, cfg, velocitymesh):
        self.backend = backend
        self.cfg = cfg
        self.vm = velocitymesh
        self.block = (256, 1, 1)

        # read model parameters
        print("\n Scattering model -----------------------------")
        self.load_parameters()

        # perform any necessary computation
        self.perform_precomputation()
        print('Finished scattering model precomputation')


    @abstractmethod
    def load_parameters(self):
        pass

    @abstractmethod
    def perform_precomputation(self):
        pass

    def swap_axes(self, d_arr_in, d_arr_out):
        raise RuntimeError("Not implemented")
        pass


"""
BGK "Iteration free" direct approach
The argument is: If the velocity grid is isotropic, and large enough; 
the error in conservation would be spectrally low
"""
class ADGFSBGKDirectGLLScatteringModel(ADGFSScatteringModel):
    scattering_model = 'bgk-direct-gll'

    def __init__(self, backend, cfg, velocitymesh, **kwargs):
        super().__init__(backend, cfg, velocitymesh, **kwargs)

    def load_parameters(self):
        Pr = 1.
        omega = self.cfg.getfloat('scattering-model', 'omega');
        muRef = self.cfg.getfloat('scattering-model', 'muRef');
        Tref = self.cfg.getfloat('scattering-model', 'Tref');

        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press

        self._prefactor = (t0*p0/visc)
        self._omega = omega
        self._Pr = Pr
        print("prefactor:", self._prefactor)

    def nu(self, rho, T):
        return rho*T**(1-self._omega);

    def perform_precomputation(self):
        self.nalph = 5
        vm = self.vm
        cv = self.vm.cv()

        # compute mat
        mat = np.vstack(
            (np.ones(vm.vsize()), 
                vm.cv(), 
                np.einsum('ij,ij->j', vm.cv(), vm.cv())
            )
        )*vm.cw() # 5 x Nv
        self.mat = self.backend.const_matrix(mat.ravel().reshape(-1,1)) # Nvx5

        # now load the modules
        self.load_modules()


    def load_modules(self):
        """Load modules (this must be called internally)"""

        self.blas = CUDACUBLASKernels() # blas kernels for computing moments

        # allocate velocity mesh in PyCUDA gpuarray
        cv = self.vm.cv()
        self.d_cvx = gpuarray.to_gpu(cv[0,:])
        self.d_cvy = gpuarray.to_gpu(cv[1,:])
        self.d_cvz = gpuarray.to_gpu(cv[2,:])

        # the precision
        dtype = np.float64
        dtypename = 'double'
        dtn = dtypename[0]

        # number of stages
        nbdf = [1, 2, 3]; nars = [1, 2, 3, 4]

        # extract the template
        dfltargs = dict(soasz=self.backend.soasz,
            vsize=self.vm.vsize(), 
            nalph=self.nalph, omega=self._omega, Pr=self._Pr,
            dtype=dtypename, nbdf=nbdf, nars=nars)
        src = DottedTemplateLookup('frfs.solvers.adgfs.kernels.scattering', 
                    dfltargs).get_template(self.scattering_model).render()
        module = compiler.SourceModule(src)

        # construct swap operation (transposes the matrix)
        self.swapKern = get_kernel(module, "swap_axes", 'iiiiPP')

        # construct maxwellian given (rho, rho*ux, rho*uy, rho*uz, E)
        self.cmaxwellianKern = get_kernel(module, "cmaxwellian", 'iiiiPPPPP')

        # update the moment
        self.updateMomKernsBDF = tuple(map(
            lambda q: get_kernel(module, "updateMom{0}_BDF".format(q), 
                dtn+'i'+dtn+(dtn+'P')*(2*q+1)+dtn), nbdf
        ))
        self.updateMomKernsARS = tuple(map(
            lambda q: get_kernel(module, "updateMom{0}_ARS".format(q), 
                dtn+'i'+dtn+dtn*(2*q)+'P'*(2*q+1)), nars
        ))

        # update the distribution
        self.updateDistKernsBDF = tuple(map(
            lambda q: get_kernel(module, "updateDistribution{0}_BDF".format(q),
                dtn+'iiii'+dtn+(dtn+'P')*(2*q+2)+'P'), nbdf
        ))
        self.updateDistKernsARS = tuple(map(
            lambda q: get_kernel(module, "updateDistribution{0}_ARS".format(q),
                dtn+'iiii'+dtn+dtn*(2*q)+'P'*(4*q+2)), nars
        ))
        self.module = module


    def swap_axes(self, fin, fout):
        nupts, ldim, _ = fin.traits
        nvars, neles = fout.ioshape[1:]

        grid_swap = get_grid_for_block(self.block, nupts*nvars*neles)
        self.swapKern.prepared_call(grid_swap, self.block, 
            nupts, ldim, nvars, neles, fin, fout)

    def moment(self, t, f, U):
        nupts, ldim, _ = f.traits
        nvars, neles = f.ioshape[1:]

        # prepare the momentum computation kernel
        sA_mom = (nupts*neles, self.vm.vsize())
        sB_mom = (self.nalph, self.vm.vsize())
        sC_mom = (nupts*neles, self.nalph)
        self.blas.mulp(
            f, sA_mom, self.mat, sB_mom, U, sC_mom)

    def constructMaxwellian(self, U, M, Ut):
        nupts, ldim, _ = M.traits
        nvars, neles = M.ioshape[1:]

        grid = get_grid_for_block(self.block, nupts*nvars*neles)
        self.cmaxwellianKern.prepared_call(grid, self.block, 
            nupts, ldim, nvars, neles, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, M, U)


    def updateMomentBDF(self, dt, *args):
        # the size of args should be 4*q+3 for BDF scheme
        q = (len(args) - 3)//4
        assert len(args)==4*q+3, "Inconsistency in number of parameters"

        lda = np.int(args[-2].ioshape[0])
        grid = get_grid_for_block(self.block, lda)
        self.updateMomKernsBDF[q-1].prepared_call(grid, self.block, 
                self._prefactor, lda, dt, *args)


    def updateDistBDF(self, dt, *args):
        # the size of args should be 4*q+5 for BDF scheme
        q = (len(args) - 5)//4
        assert len(args)==4*q+5, "Inconsistency in number of parameters"

        nupts, ldim, _ = args[1].traits
        nvars, neles = args[1].ioshape[1:]
        grid = get_grid_for_block(self.block, nupts*nvars*neles)
        self.updateDistKernsBDF[q-1].prepared_call(grid, self.block, 
            self._prefactor, nupts, ldim, nvars, neles, 
            dt, *args)


    def updateMomentARS(self, dt, *args):
        # the size of args should be 4*q+1 for ARS scheme
        q = (len(args) - 1)//4
        assert len(args)==4*q+1, "Inconsistency in number of parameters"

        lda = np.int(args[-1].ioshape[0])
        grid = get_grid_for_block(self.block, lda)
        self.updateMomKernsARS[q-1].prepared_call(grid, self.block, 
                self._prefactor, lda, dt, *args)


    def updateDistARS(self, dt, *args):
        # the size of args should be 6*q+2 for ARS scheme
        q = (len(args) - 2)//6
        assert len(args)==6*q+2, "Inconsistency in number of parameters"

        nupts, ldim, _ = args[-1].traits
        nvars, neles = args[-1].ioshape[1:]
        grid = get_grid_for_block(self.block, nupts*nvars*neles)
        self.updateDistKernsARS[q-1].prepared_call(grid, self.block, 
            self._prefactor, nupts, ldim, nvars, neles, 
            dt, *args)




"""
ESBGK "Iteration free" direct approach
The argument is: If the velocity grid is isotropic, and large enough; 
the error in conservation would be spectrally low
"""
class ADGFSESBGKDirectGLLScatteringModel(ADGFSBGKDirectGLLScatteringModel):
    scattering_model = 'esbgk-direct-gll'

    def load_parameters(self):
        Pr = self.cfg.getfloat('scattering-model', 'Pr', 2./3.);
        omega = self.cfg.getfloat('scattering-model', 'omega');
        muRef = self.cfg.getfloat('scattering-model', 'muRef');
        Tref = self.cfg.getfloat('scattering-model', 'Tref');

        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press

        self._prefactor = (t0*Pr*p0/visc)
        self._omega = omega
        self._Pr = Pr
        print("prefactor:", self._prefactor)


    def perform_precomputation(self):
        self.nalph = 11
        vm = self.vm
        cv = vm.cv()

        # compute mat
        mat = np.vstack(
            (np.ones(vm.vsize()), # mass
            cv, # momentum
            np.einsum('ij,ij->j', vm.cv(), vm.cv()), # energy
            cv[0,:]*cv[0,:], cv[1,:]*cv[1,:], cv[2,:]*cv[2,:], # normal stress
            cv[0,:]*cv[1,:], cv[1,:]*cv[2,:], cv[2,:]*cv[0,:] # off-diag 
        ))*vm.cw() # 11 x Nv
        self.mat = self.backend.const_matrix(mat.ravel().reshape(-1,1)) # Nv x 11 flatenned

        # now load the modules
        self.load_modules()

        # construct (rho, u, T, P) given (rho, rho*u, E, P)
        self.momentNormKern = get_kernel(self.module, "momentNorm", 'iPP')


    def constructMaxwellian(self, U, M, Ut):
        nupts, ldim, _ = M.traits
        nvars, neles = M.ioshape[1:]

        grid = get_grid_for_block(self.block, U.ioshape[0])
        self.momentNormKern.prepared_call(grid, self.block, U.ioshape[0], U, Ut)

        grid = get_grid_for_block(self.block, nupts*nvars*neles)
        self.cmaxwellianKern.prepared_call(grid, self.block, 
            nupts, ldim, nvars, neles, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, M, Ut)



"""
Shakov "Iteration free" direct approach
The argument is: If the velocity grid is isotropic, and large enough; 
the error in conservation would be spectrally low
"""
class ADGFSShakovDirectGLLScatteringModel(ADGFSESBGKDirectGLLScatteringModel):
    scattering_model = 'shakov-direct-gll'

    def load_parameters(self):
        Pr = self.cfg.getfloat('scattering-model', 'Pr', 2./3.);
        omega = self.cfg.getfloat('scattering-model', 'omega');
        muRef = self.cfg.getfloat('scattering-model', 'muRef');
        Tref = self.cfg.getfloat('scattering-model', 'Tref');

        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press

        self._prefactor = (t0*p0/visc)
        self._omega = omega
        self._Pr = Pr
        print("prefactor:", self._prefactor)


    def perform_precomputation(self):
        self.nalph = 14
        vm = self.vm
        cv = vm.cv()

        # compute mat
        mat = np.vstack(
            (np.ones(vm.vsize()), # mass
            cv, # momentum
            np.einsum('ij,ij->j', vm.cv(), vm.cv()), # energy
            cv[0,:]*cv[0,:], cv[1,:]*cv[1,:], cv[2,:]*cv[2,:], # normal stress
            cv[0,:]*cv[1,:], cv[1,:]*cv[2,:], cv[2,:]*cv[0,:], # off-diag 
            np.einsum('ij,ij->j', cv, cv)*cv[0,:], # x-heat-flux
            np.einsum('ij,ij->j', cv, cv)*cv[1,:], # y-heat-flux
            np.einsum('ij,ij->j', cv, cv)*cv[2,:] # z-heat-flux
        ))*vm.cw() # 14 x Nv
        self.mat = self.backend.const_matrix(mat.ravel().reshape(-1,1)) # Nvx11

        # now load the modules
        self.load_modules()

        # construct (rho, u, T, P) given (rho, rho*u, E, P)
        self.momentNormKern = get_kernel(self.module, "momentNorm", 'iPP')

        
