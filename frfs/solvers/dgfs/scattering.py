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

class DGFSScatteringModel(object, metaclass=ABCMeta):
    def __init__(self, backend, cfg, velocitymesh):
        self.backend = backend
        self.cfg = cfg
        self.vm = velocitymesh

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

    @abstractmethod 
    def fs(self, d_arr_in, d_arr_out, elem, upt):
        pass

# for gll
class DGFSVHSGLLScatteringModel(DGFSScatteringModel):
    scattering_model = 'vhs-gll'

    def __init__(self, backend, cfg, velocitymesh):
        super().__init__(backend, cfg, velocitymesh)

    def load_parameters(self):
        alpha = 1.0
        omega = self.cfg.getfloat('scattering-model', 'omega');
        self._gamma = 2.0*(1-omega)

        dRef = self.cfg.getfloat('scattering-model', 'dRef');
        Tref = self.cfg.getfloat('scattering-model', 'Tref');

        invKn = self.vm.H0()*np.sqrt(2.0)*np.pi*self.vm.n0()*dRef*dRef*pow(
            Tref/self.vm.T0(), omega-0.5);

        self._prefactor = invKn*alpha/(
            pow(2.0, 2-omega+alpha)*gamma(2.5-omega)*np.pi);

        print("Kn:", 1.0/invKn)
        print("prefactor:", self._prefactor)

    def perform_precomputation(self):
        # Precompute aa, bb1, bb2 (required for kernel)
        # compute l
        Nv = self.vm.Nv()
        Nrho = self.vm.Nrho()
        M = self.vm.M()
        L = self.vm.L()
        qz = self.vm.qz()
        qw = self.vm.qw()
        sz = self.vm.sz()
        vsize = self.vm.vsize()

        l0 = np.concatenate((np.arange(0,Nv/2), np.arange(-Nv/2, 0)))
        #l = l0[np.mgrid[0:Nv, 0:Nv, 0:Nv]]
        #l = l.reshape((3,vsize))
        l = np.zeros((3,vsize))
        for idv in range(vsize):
            I = int(idv/(Nv*Nv))
            J = int((idv%(Nv*Nv))/Nv)
            K = int((idv%(Nv*Nv))%Nv)
            l[0,idv] = l0[I];
            l[1,idv] = l0[J];
            l[2,idv] = l0[K];
        d_lx = gpuarray.to_gpu(np.ascontiguousarray(l[0,:]))
        d_ly = gpuarray.to_gpu(np.ascontiguousarray(l[1,:]))
        d_lz = gpuarray.to_gpu(np.ascontiguousarray(l[2,:]))

        """
        # the vars
        h_aa = np.zeros(Nrho*M*vsize)
        h_bb1 = np.zeros(Nrho*vsize)
        h_bb2 = np.zeros(vsize)

        # Final precomputation
        # TODO: transform this to matrix computations
        for p in range(Nrho):
            for q in range(M):
                for r in range(vsize):
                    h_aa[ (p*M+q)*vsize + r] = (
                        np.pi/L*qz[p]/2.0*np.dot(l[:,r], sz[q,:])
                    );

            # sinc function has different defn in numpy
            for r in range(vsize):
                h_bb1[p*vsize+r] = pow(qz[p],(self._gamma+2))*4*np.pi*(
                    np.sinc(1./L*qz[p]/2.0*np.sqrt(np.dot(l[:,r],l[:,r]))))

                h_bb2[r] += qw[p]*pow(qz[p],(self._gamma+2))*16*np.pi*np.pi*(
                          np.sinc(1./L*qz[p]*np.sqrt(np.dot(l[:,r],l[:,r]))))

        # load precomputed weights onto the memory
        #self.d_qw = gpuarray.to_gpu(qw)
        
        # load precomputed weights onto the memory
        self.d_aa = gpuarray.to_gpu(h_aa)
        self.d_bb1 = gpuarray.to_gpu(h_bb1)
        self.d_bb2 = gpuarray.to_gpu(h_bb2)

        #d_Q = gpuarray.empty_like(d_f0)
        """
        dtype = np.float64

        # define scratch  spaces
        self.d_FTf = gpuarray.empty(vsize, dtype=np.complex128)
        self.d_fC = gpuarray.empty_like(self.d_FTf)
        self.d_QG = gpuarray.empty_like(self.d_FTf)
        self.d_t1 = gpuarray.empty(M*Nrho*vsize, dtype=np.complex128)
        self.d_t2 = gpuarray.empty_like(self.d_t1)
        self.d_t3 = gpuarray.empty_like(self.d_t1)
        self.d_t4 = gpuarray.empty_like(self.d_t1)

        self.block = (256, 1, 1)
        self.grid = get_grid_for_block(self.block, vsize)

        # define complex to complex plan
        rank = 3
        n = np.array([Nv, Nv, Nv], dtype=np.int32)

        #planD2Z = cufftPlan3d(Nv, Nv, Nv, CUFFT_D2Z)
        self.planZ2Z_MNrho = cufftPlanMany(rank, n.ctypes.data,
            None, 1, vsize, 
            None, 1, vsize, 
            CUFFT_Z2Z, M*Nrho)
        self.planZ2Z = cufftPlan3d(Nv, Nv, Nv, CUFFT_Z2Z)

        dfltargs = dict(Nrho=Nrho, M=M, 
            vsize=vsize, sw=self.vm.sw(), prefac=self._prefactor, 
            soasz=self.backend.soasz, qw=qw, sz=sz, gamma=self._gamma, 
            L=L, qz=qz)
        src = DottedTemplateLookup(
            'frfs.solvers.dgfs.kernels.scattering', dfltargs
        ).get_template('vhs-gll').render()

        # Compile the source code and retrieve the kernel
        module = compiler.SourceModule(src)

        self.d_aa = gpuarray.empty(Nrho*M*vsize, dtype=dtype)
        precompute_aa = module.get_function("precompute_aa")
        precompute_aa.prepare('PPPP')
        precompute_aa.set_cache_config(cuda.func_cache.PREFER_L1)
        precompute_aa.prepared_call(self.grid, self.block, d_lx.ptr, d_ly.ptr, 
            d_lz.ptr, self.d_aa.ptr)

        self.d_bb1 = gpuarray.empty(Nrho*vsize, dtype=dtype)
        self.d_bb2 = gpuarray.empty(vsize, dtype=dtype)
        precompute_bb = module.get_function("precompute_bb")
        precompute_bb.prepare('PPPPP')
        precompute_bb.set_cache_config(cuda.func_cache.PREFER_L1)
        precompute_bb.prepared_call(self.grid, self.block, d_lx.ptr, d_ly.ptr, 
            d_lz.ptr, self.d_bb1.ptr, self.d_bb2.ptr)

        # transform scalar to complex
        self.r2zKern = module.get_function("r2z")
        self.r2zKern.prepare('IIIIIIPP')
        self.r2zKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the cosSinMul kernel for execution
        self.cosSinMultKern = module.get_function("cosSinMul")
        self.cosSinMultKern.prepare('PPPP')
        self.cosSinMultKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the magSqrKern kernel for execution
        self.magSqrKern = module.get_function("magSqr")
        self.magSqrKern.prepare('PPP')
        self.magSqrKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the computeQG kernel for execution
        self.computeQGKern = module.get_function("computeQG")
        self.computeQGKern.prepare('PPP')
        self.computeQGKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the ax kernel for execution
        self.axKern = module.get_function("ax")
        self.axKern.prepare('PP')
        self.axKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the real kernel for execution
        #self.realKern = module.get_function("real")
        #self.realKern.prepare('IIIIIIPPP')
        #self.realKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the out kernel for execution
        self.outKern = module.get_function("output")
        self.outKern.prepare('IIIIIIPPPP')
        self.outKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the scale kernel for execution
        #self.scaleKern = module.get_function("scale")
        #self.scaleKern.prepare('P')
        #self.scaleKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the scale kernel for execution
        #self.scaleMNKern = module.get_function("scale_MN")
        #self.scaleMNKern.prepare('P')
        #self.scaleMNKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # required by the child class (may be deleted by the child)
        self.module = module

        
    def fs(self, d_arr_in, d_arr_out, elem, upt):
        nrow, ldim, _ = d_arr_in.traits
        ncola, ncolb = d_arr_in.ioshape[1:]

        d_f0 = d_arr_in._as_parameter_
        d_Q = d_arr_out._as_parameter_
            
        # construct d_fC from d_f0
        self.r2zKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, d_f0, self.d_fC.ptr)

        # compute forward FFT of f | Ftf = fft(f)
        cufftExecZ2Z(self.planZ2Z, 
            self.d_fC.ptr, self.d_FTf.ptr, CUFFT_FORWARD)
        #self.scaleKern.prepared_call(self.grid, self.block, 
        #    self.d_FTf.ptr)
        
        # compute t1_{pqr} = cos(a_{pqr})*FTf_r; t2_{pqr} = sin(a_{pqr})*FTf_r
        # scales d_FTf
        self.cosSinMultKern.prepared_call(self.grid, self.block, 
            self.d_aa.ptr, self.d_FTf.ptr, self.d_t1.ptr, self.d_t2.ptr)

        # compute inverse fft 
        cufftExecZ2Z(self.planZ2Z_MNrho, 
            self.d_t1.ptr, self.d_t3.ptr, CUFFT_INVERSE)
        cufftExecZ2Z(self.planZ2Z_MNrho, 
            self.d_t2.ptr, self.d_t4.ptr, CUFFT_INVERSE)

        # compute t2 = t3^2 + t4^2
        self.magSqrKern.prepared_call(self.grid, self.block, 
            self.d_t3.ptr, self.d_t4.ptr, self.d_t2.ptr)

        # compute t1 = fft(t2)
        cufftExecZ2Z(self.planZ2Z_MNrho, 
            self.d_t2.ptr, self.d_t1.ptr, CUFFT_FORWARD)
        # scaling factor is multiplied in the computeQGKern 
        # note: t1 is not modified in computeQGKern
        #self.scaleMNKern.prepared_call(self.grid, self.block, 
        #    self.d_t1.ptr)

        # compute fC_r = 2*wrho_p*ws*b1_p*t1_r
        self.computeQGKern.prepared_call(self.grid, self.block, 
            self.d_bb1.ptr, self.d_t1.ptr, self.d_fC.ptr)

        # inverse fft| QG = iff(fC)  [Gain computed]
        cufftExecZ2Z(self.planZ2Z, 
            self.d_fC.ptr, self.d_QG.ptr, CUFFT_INVERSE)

        # compute FTf_r = b2_r*FTf_r
        self.axKern.prepared_call(self.grid, self.block, 
            self.d_bb2.ptr, self.d_FTf.ptr)

        # inverse fft| fC = iff(FTf)
        cufftExecZ2Z(self.planZ2Z, 
            self.d_FTf.ptr, self.d_fC.ptr, CUFFT_INVERSE)
        
        # outKern
        self.outKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, 
            self.d_QG.ptr, self.d_fC.ptr, d_f0, d_Q)
        



# for full nodal version
class DGFSVHSGLLNodalScatteringModel(DGFSVHSGLLScatteringModel):
    scattering_model = 'vhs-gll-nodal'

    def __init__(self, backend, cfg, velocitymesh):
        super().__init__(backend, cfg, velocitymesh)

    def perform_precomputation(self):
        super().perform_precomputation()

        self.d_FTg = gpuarray.empty_like(self.d_FTf)
        self.d_gC = gpuarray.empty_like(self.d_FTf)
        self.d_t5 = gpuarray.empty_like(self.d_t1)

        # Prepare the cosMul kernel for execution
        self.cosMultKern = self.module.get_function("cosMul")
        self.cosMultKern.prepare('PPPPP')
        self.cosMultKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the cosMul kernel for execution
        self.sinMultKern = self.module.get_function("sinMul")
        self.sinMultKern.prepare('PPPPP')
        self.sinMultKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the cplxMul kernel for execution
        self.cplxMul = self.module.get_function("cplxMul")
        self.cplxMul.prepare('PPP')
        self.cplxMul.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the cplxMulAdd kernel for execution
        self.cplxMulAdd = self.module.get_function("cplxMulAdd")
        self.cplxMulAdd.prepare('PPP')
        self.cplxMulAdd.set_cache_config(cuda.func_cache.PREFER_L1)

        del self.module

    def fs(self, d_arr_in1, d_arr_in2, d_arr_out, elem, upt):
        nrow, ldim, _ = d_arr_in1.traits
        ncola, ncolb = d_arr_in1.ioshape[1:]

        d_f = d_arr_in1._as_parameter_
        d_g = d_arr_in2._as_parameter_
        d_Q = d_arr_out._as_parameter_
            
        # construct d_fC from d_f
        self.r2zKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, d_f, self.d_fC.ptr)

        # construct d_gC from d_g
        self.r2zKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, d_g, self.d_gC.ptr)

        # compute forward FFT of f | FTf = fft(f)
        cufftExecZ2Z(self.planZ2Z, 
            self.d_fC.ptr, self.d_FTf.ptr, CUFFT_FORWARD)
        #self.scaleKern.prepared_call(self.grid, self.block, 
        #    self.d_FTf.ptr)

        # compute forward FFT of g | FTg = fft(g)
        cufftExecZ2Z(self.planZ2Z, 
            self.d_gC.ptr, self.d_FTg.ptr, CUFFT_FORWARD)
        #self.scaleKern.prepared_call(self.grid, self.block, 
        #    self.d_FTg.ptr)
        
        # compute t1_{pqr} = cos(a_{pqr})*FTf_r; t2_{pqr} = cos(a_{pqr})*FTg_r
        # scales d_FTf, d_FTg
        self.cosMultKern.prepared_call(self.grid, self.block, 
            self.d_aa.ptr, self.d_FTf.ptr, self.d_FTg.ptr, 
            self.d_t1.ptr, self.d_t2.ptr)

        # compute inverse fft 
        cufftExecZ2Z(self.planZ2Z_MNrho, 
            self.d_t1.ptr, self.d_t3.ptr, CUFFT_INVERSE)
        cufftExecZ2Z(self.planZ2Z_MNrho, 
            self.d_t2.ptr, self.d_t4.ptr, CUFFT_INVERSE)

        # compute t5 = t3*t4
        self.cplxMul.prepared_call(self.grid, self.block, 
            self.d_t3.ptr, self.d_t4.ptr, self.d_t5.ptr)

        # compute t1_{pqr} = sin(a_{pqr})*FTf_r; t2_{pqr} = sin(a_{pqr})*FTg_r
        # "does not" scale d_FTf, d_FTg
        self.sinMultKern.prepared_call(self.grid, self.block, 
            self.d_aa.ptr, self.d_FTf.ptr, self.d_FTg.ptr, 
            self.d_t1.ptr, self.d_t2.ptr)

        # compute inverse fft 
        cufftExecZ2Z(self.planZ2Z_MNrho, 
            self.d_t1.ptr, self.d_t3.ptr, CUFFT_INVERSE)
        cufftExecZ2Z(self.planZ2Z_MNrho, 
            self.d_t2.ptr, self.d_t4.ptr, CUFFT_INVERSE)

        # compute t5 += t3*t4
        self.cplxMulAdd.prepared_call(self.grid, self.block, 
            self.d_t3.ptr, self.d_t4.ptr, self.d_t5.ptr)

        # compute t1 = fft(t2)
        cufftExecZ2Z(self.planZ2Z_MNrho, 
            self.d_t5.ptr, self.d_t1.ptr, CUFFT_FORWARD)
        # scaling factor is multiplied in the computeQGKern 
        # note: t1 is not modified in computeQGKern
        #self.scaleMNKern.prepared_call(self.grid, self.block, 
        #    self.d_t1.ptr)

        # compute fC_r = 2*wrho_p*ws*b1_p*t1_r
        self.computeQGKern.prepared_call(self.grid, self.block, 
            self.d_bb1.ptr, self.d_t1.ptr, self.d_fC.ptr)

        # inverse fft| QG = iff(fC)  [Gain computed]
        cufftExecZ2Z(self.planZ2Z, 
            self.d_fC.ptr, self.d_QG.ptr, CUFFT_INVERSE)

        # compute FTg_r = b2_r*FTg_r
        self.axKern.prepared_call(self.grid, self.block, 
            self.d_bb2.ptr, self.d_FTg.ptr)

        # inverse fft| fC = iff(FTg)
        cufftExecZ2Z(self.planZ2Z, 
            self.d_FTg.ptr, self.d_fC.ptr, CUFFT_INVERSE)
        
        # outKern
        self.outKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, 
            self.d_QG.ptr, self.d_fC.ptr, d_f, d_Q)















# for BGK scattering model
"""
Reference:
Sruti Chigullapalli, PhD thesis, 2011, Purdue University
Deterministic Approach for Unsteady Rarefied Flow Simulations in Complex
Geometries and its Application to Gas Flows in Microsystems.
"""
class DGFSBGKGLLScatteringModel(DGFSScatteringModel):
    scattering_model = 'bgk-gll'

    def __init__(self, backend, cfg, velocitymesh):
        super().__init__(backend, cfg, velocitymesh)

    def load_parameters(self):
        Pr = 1.
        omega = self.cfg.getfloat('scattering-model', 'omega');
        muRef = self.cfg.getfloat('scattering-model', 'muRef');
        Tref = self.cfg.getfloat('scattering-model', 'Tref');

        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press

        self._prefactor = (t0*Pr*p0/visc)
        self._omega = omega
        print("prefactor:", self._prefactor)

    def perform_precomputation(self):
        # Precompute aa, bb1, bb2 (required for kernel)
        # compute l
        Nv = self.vm.Nv()
        Nrho = self.vm.Nrho()
        M = self.vm.M()
        L = self.vm.L()
        qz = self.vm.qz()
        qw = self.vm.qw()
        sz = self.vm.sz()
        vsize = self.vm.vsize()
        cv = self.vm.cv()
        self.cw = self.vm.cw()

        # the precision
        dtype = np.float64
        d = 'd'

        # number of alpha variables to be determined 
        self.nalph = 5
        nalph = self.nalph

        # allocate velocity mesh in PyCUDA gpuarray
        self.d_cvx = gpuarray.to_gpu(cv[0,:])
        self.d_cvy = gpuarray.to_gpu(cv[1,:])
        self.d_cvz = gpuarray.to_gpu(cv[2,:])
        self.d_cSqr = gpuarray.to_gpu(np.einsum('ij,ij->j', cv, cv))

        # define scratch spaces
        # cell local distribution function
        self.d_floc = gpuarray.empty(vsize, dtype=dtype)

        # cell local equilibrium distribution function
        self.d_fe = gpuarray.empty_like(self.d_floc)         

        # scratch variable for storage for the moments
        self.d_mom10 = gpuarray.empty_like(self.d_floc)
        self.d_mom11 = gpuarray.empty_like(self.d_floc)
        self.d_mom12 = gpuarray.empty_like(self.d_floc)
        self.d_mom2 = gpuarray.empty_like(self.d_floc)

        # storage for reduced moments 
        self.d_moms = [gpuarray.empty(1,dtype=dtype) for i in range(nalph)]
        self.d_equiMoms = [
            gpuarray.empty(1,dtype=dtype) for i in range(nalph)]

        # compute mBGK
        mBGK = np.vstack(
            (np.ones(vsize), cv, np.einsum('ij,ij->j', cv, cv))
        ) # 5 x vsize
        self.d_mBGK = gpuarray.to_gpu((mBGK).ravel()) # vsize x 5 flatenned

        # storage for expM
        self.d_expM = gpuarray.empty_like(self.d_mBGK)

        # block size for running the kernel
        self.block = (256, 1, 1)
        self.grid = get_grid_for_block(self.block, vsize)

        # storage for alpha
        self.d_alpha = gpuarray.empty(nalph, dtype=dtype)

        # residual 
        self.d_res = gpuarray.empty(1, dtype=dtype)

        # for defining ptr
        self.ptr = lambda x: list(map(lambda v: v.ptr, x))

        # extract the template
        dfltargs = dict(cw=self.cw,
            vsize=vsize, prefac=self._prefactor, 
            soasz=self.backend.soasz, nalph=self.nalph, omega=self._omega,
            block_size=self.block[0])
        src = DottedTemplateLookup(
            'frfs.solvers.dgfs.kernels.scattering', dfltargs
        ).get_template('bgk-gll').render()

        # Compile the source code and retrieve the kernel
        module = compiler.SourceModule(src)

        # sum kernel
        self.sumKern = module.get_function("sum_")
        self.sumKern.prepare('PPII')
        self.sumKern.set_cache_config(cuda.func_cache.PREFER_L1)
        grid = self.grid
        block = self.block
        seq_count0 = int(4)
        N = int(vsize)
        #grid_count0 = int((grid[0] + (-grid[0] % seq_count0)) // seq_count0)
        grid_count0 = int((grid[0]//seq_count0 + ceil(grid[0]%seq_count0)))
        d_st = gpuarray.empty(grid_count0, dtype=dtype)
        #seq_count1 = int((grid_count0 + (-grid_count0 % block[0])) // block[0])
        seq_count1 = int((grid_count0//block[0] + ceil(grid_count0%block[0])))
        def sum_(d_in, d_out):
            self.sumKern.prepared_call(
                (grid_count0,1), block, d_in.ptr, d_st.ptr, seq_count0, N)
            self.sumKern.prepared_call(
                (1,1), block, d_st.ptr, d_out.ptr, seq_count1, grid_count0)
        self.sumFunc = sum_

        # extract the element-local space coefficients
        self.flocKern = module.get_function("flocKern")
        self.flocKern.prepare('IIIIIIPP')
        self.flocKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # compute the first moment of local distribution
        self.mom1Kern = module.get_function("mom1")
        self.mom1Kern.prepare('PPP'+'P'+'PPP')
        self.mom1Kern.set_cache_config(cuda.func_cache.PREFER_L1)

        # helper function for recovering density/velocity from moments
        self.mom01NormKern = module.get_function("mom01Norm")
        self.mom01NormKern.prepare('PPPP')
        self.mom01NormKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # compute the second moment of local distribution
        self.mom2Kern = module.get_function("mom2")
        self.mom2Kern.prepare('PPP'+'PP'+'P'*4)
        self.mom2Kern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the equiDistInit kernel for execution
        self.equiDistInitKern = module.get_function("equiDistInit")
        self.equiDistInitKern.prepare('P'+'P'*nalph)
        self.equiDistInitKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # compute the moments of equilibrium distribution
        self.equiDistMomKern = module.get_function("equiDistMom")
        self.equiDistMomKern.prepare('PPPP'+'P'+'PPPP')
        self.equiDistMomKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the equiDistCompute kernel for execution
        self.equiDistComputeKern = module.get_function("equiDistCompute")
        self.equiDistComputeKern.prepare('PPPP'+'PPP'+'P'*3)
        self.equiDistComputeKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the gaussElim kernel for execution
        self.gaussElimKern = module.get_function("gaussElim")
        self.gaussElimKern.prepare('PP'+'P'*(nalph*2+1))
        self.gaussElimKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the out kernel for execution
        self.outKern = module.get_function("output")
        self.outKern.prepare('IIIIII'+'P'*2+'PPP')
        self.outKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # required by the child class (may be deleted by the child)
        self.module = module

        # define a blas handle
        self.blas = CUDACUBLASKernels()

        # multiplication kernel "specifically" for computing jacobian
        sA = (nalph, vsize)
        sB = (nalph, vsize)
        sC = (nalph, nalph)        
        self.jacMulFunc = lambda A, B, C: self.blas.mul(A, sA, B, sB, C, sC)
        self.d_J = gpuarray.empty(nalph*nalph, dtype=dtype)
        

    def fs(self, d_arr_in, d_arr_out, elem, upt):
        nrow, ldim, _ = d_arr_in.traits
        ncola, ncolb = d_arr_in.ioshape[1:]

        d_f0 = d_arr_in._as_parameter_
        d_Q = d_arr_out._as_parameter_

        # construct d_floc from d_f0
        self.flocKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, d_f0, self.d_floc.ptr)

        # compute first moment
        self.mom1Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, self.d_floc.ptr,
            self.d_mom10.ptr, self.d_mom11.ptr, self.d_mom12.ptr)

        # compute macroscopic properties
        self.sumFunc(self.d_floc, self.d_moms[0])  # missing: ${cw}
        self.sumFunc(self.d_mom10, self.d_moms[1]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom11, self.d_moms[2]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom12, self.d_moms[3]) # missing: ${cw}/d_moms[0]

        # insert the missing factor for the density and velocity
        self.mom01NormKern.prepared_call((1,1), (1,1,1), 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the second moments
        self.mom2Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_floc.ptr, self.d_mom2.ptr, 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the temperature: missing factor cw/(1.5*d_moms[0])
        self.sumFunc(self.d_mom2, self.d_moms[4])

        # we compute alpha (initial guess)
        #alpha[0], alpha[1] = locRho/((np.pi*locT)**1.5), 1./locT
        # the missing factor in d_moms[4] is added here
        self.equiDistInitKern.prepared_call((1,1), (1,1,1),
            self.d_alpha.ptr, *self.ptr(self.d_moms))

        # initialize the residual and the tolerances
        res, initRes, iterTol = 1.0, 1.0, 1e-10
        nIter, maxIters = 0, 20

        # start the iteration        
        while res/initRes>iterTol:

            self.equiDistComputeKern.prepared_call(self.grid, self.block,
                self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
                self.d_alpha.ptr,
                self.d_floc.ptr, self.d_fe.ptr, self.d_expM.ptr,
                self.d_moms[1].ptr, self.d_moms[2].ptr, self.d_moms[3].ptr)

            # form the jacobian 
            self.jacMulFunc(self.d_mBGK, self.d_expM, self.d_J)

            # compute moments (gemv/gemm is slower)
            self.equiDistMomKern.prepared_call(self.grid, self.block, 
                self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, 
                self.d_cSqr.ptr, self.d_fe.ptr, 
                self.d_mom10.ptr, self.d_mom11.ptr, 
                self.d_mom12.ptr, self.d_mom2.ptr)
            self.sumFunc(self.d_fe, self.d_equiMoms[0])
            self.sumFunc(self.d_mom10, self.d_equiMoms[1])
            self.sumFunc(self.d_mom11, self.d_equiMoms[2])
            self.sumFunc(self.d_mom12, self.d_equiMoms[3])
            self.sumFunc(self.d_mom2, self.d_equiMoms[4])

            self.gaussElimKern.prepared_call((1,1), (1,1,1),
                self.d_res.ptr, self.d_alpha.ptr,
                *self.ptr(list(self.d_moms + self.d_equiMoms + [self.d_J]))
            )

            res = self.d_res.get()[0] #sum(abs(F))
            if nIter==0: initRes = res
            if(isnan(res)): raise RuntimeError("NaN encountered")

            # increment iterations
            nIter += 1

            # break if the number of iterations are greater
            if nIter>maxIters: break

        # outKern
        self.outKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, 
            self.d_moms[0].ptr, self.d_moms[4].ptr, 
            self.d_fe.ptr, self.d_floc.ptr, d_Q)




















"""
BGK "Iteration free" direct approach
The argument is: If the velocity grid is isotropic, and large enough; 
the error in conservation would be spectrally low
"""
class DGFSBGKDirectGLLScatteringModel(DGFSScatteringModel):
    scattering_model = 'bgk-direct-gll'

    def __init__(self, backend, cfg, velocitymesh):
        super().__init__(backend, cfg, velocitymesh)

    def load_parameters(self):
        Pr = 1.
        omega = self.cfg.getfloat('scattering-model', 'omega');
        muRef = self.cfg.getfloat('scattering-model', 'muRef');
        Tref = self.cfg.getfloat('scattering-model', 'Tref');

        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press

        self._prefactor = (t0*Pr*p0/visc)
        self._omega = omega
        print("prefactor:", self._prefactor)

    def perform_precomputation(self):
        # Precompute aa, bb1, bb2 (required for kernel)
        # compute l
        Nv = self.vm.Nv()
        Nrho = self.vm.Nrho()
        M = self.vm.M()
        L = self.vm.L()
        qz = self.vm.qz()
        qw = self.vm.qw()
        sz = self.vm.sz()
        vsize = self.vm.vsize()
        cv = self.vm.cv()
        self.cw = self.vm.cw()

        # the precision
        dtype = np.float64
        d = 'd'

        # number of alpha variables to be determined 
        self.nalph = 5
        nalph = self.nalph

        # allocate velocity mesh in PyCUDA gpuarray
        self.d_cvx = gpuarray.to_gpu(cv[0,:])
        self.d_cvy = gpuarray.to_gpu(cv[1,:])
        self.d_cvz = gpuarray.to_gpu(cv[2,:])
        self.d_cSqr = gpuarray.to_gpu(np.einsum('ij,ij->j', cv, cv))

        # define scratch spaces
        # cell local distribution function
        self.d_floc = gpuarray.empty(vsize, dtype=dtype)

        # cell local equilibrium distribution function
        self.d_fe = gpuarray.empty_like(self.d_floc)         

        # scratch variable for storage for the moments
        self.d_mom10 = gpuarray.empty_like(self.d_floc)
        self.d_mom11 = gpuarray.empty_like(self.d_floc)
        self.d_mom12 = gpuarray.empty_like(self.d_floc)
        self.d_mom2 = gpuarray.empty_like(self.d_floc)

        # storage for reduced moments 
        self.d_moms = [gpuarray.empty(1,dtype=dtype) for i in range(nalph)]

        # block size for running the kernel
        self.block = (256, 1, 1)
        self.grid = get_grid_for_block(self.block, vsize)

        # for defining ptr
        self.ptr = lambda x: list(map(lambda v: v.ptr, x))

        # extract the template
        dfltargs = dict(cw=self.cw,
            vsize=vsize, prefac=self._prefactor, 
            soasz=self.backend.soasz, nalph=self.nalph, omega=self._omega,
            block_size=self.block[0])
        src = DottedTemplateLookup(
            'frfs.solvers.dgfs.kernels.scattering', dfltargs
        ).get_template('bgk-direct-gll').render()

        # Compile the source code and retrieve the kernel
        module = compiler.SourceModule(src)

        # sum kernel
        self.sumKern = module.get_function("sum_")
        self.sumKern.prepare('PPII')
        self.sumKern.set_cache_config(cuda.func_cache.PREFER_L1)
        grid = self.grid
        block = self.block
        seq_count0 = int(4)
        N = int(vsize)
        #grid_count0 = int((grid[0] + (-grid[0] % seq_count0)) // seq_count0)
        grid_count0 = int((grid[0]//seq_count0 + ceil(grid[0]%seq_count0)))
        d_st = gpuarray.empty(grid_count0, dtype=dtype)
        #seq_count1 = int((grid_count0 + (-grid_count0 % block[0])) // block[0])
        seq_count1 = int((grid_count0//block[0] + ceil(grid_count0%block[0])))
        def sum_(d_in, d_out):
            self.sumKern.prepared_call(
                (grid_count0,1), block, d_in.ptr, d_st.ptr, seq_count0, N)
            self.sumKern.prepared_call(
                (1,1), block, d_st.ptr, d_out.ptr, seq_count1, grid_count0)
        self.sumFunc = sum_

        # extract the element-local space coefficients
        self.flocKern = module.get_function("flocKern")
        self.flocKern.prepare('IIIIIIPP')
        self.flocKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # compute the first moment of local distribution
        self.mom1Kern = module.get_function("mom1")
        self.mom1Kern.prepare('PPP'+'P'+'PPP')
        self.mom1Kern.set_cache_config(cuda.func_cache.PREFER_L1)

        # helper function for recovering density/velocity from moments
        self.mom01NormKern = module.get_function("mom01Norm")
        self.mom01NormKern.prepare('PPPP')
        self.mom01NormKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # compute the second moment of local distribution
        self.mom2Kern = module.get_function("mom2")
        self.mom2Kern.prepare('PPP'+'PP'+'P'*4)
        self.mom2Kern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the equiDistInit kernel for execution
        self.equiDistInitKern = module.get_function("equiDistInit")
        self.equiDistInitKern.prepare('P'*nalph)
        self.equiDistInitKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the equiDistCompute kernel for execution
        self.equiDistComputeKern = module.get_function("equiDistCompute")
        self.equiDistComputeKern.prepare('PPPP'+'P'*nalph)
        self.equiDistComputeKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the out kernel for execution
        self.outKern = module.get_function("output")
        self.outKern.prepare('IIIIII'+'P'*2+'PPP')
        self.outKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # required by the child class (may be deleted by the child)
        self.module = module


    def fs(self, d_arr_in, d_arr_out, elem, upt):
        nrow, ldim, _ = d_arr_in.traits
        ncola, ncolb = d_arr_in.ioshape[1:]

        d_f0 = d_arr_in._as_parameter_
        d_Q = d_arr_out._as_parameter_

        # construct d_floc from d_f0
        self.flocKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, d_f0, self.d_floc.ptr)

        # compute first moment
        self.mom1Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, self.d_floc.ptr,
            self.d_mom10.ptr, self.d_mom11.ptr, self.d_mom12.ptr)

        # compute macroscopic properties
        self.sumFunc(self.d_floc, self.d_moms[0])  # missing: ${cw}
        self.sumFunc(self.d_mom10, self.d_moms[1]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom11, self.d_moms[2]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom12, self.d_moms[3]) # missing: ${cw}/d_moms[0]

        # insert the missing factor for the density and velocity
        self.mom01NormKern.prepared_call((1,1), (1,1,1), 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the second moments
        self.mom2Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_floc.ptr, self.d_mom2.ptr, 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the temperature: missing factor cw/(1.5*d_moms[0])
        self.sumFunc(self.d_mom2, self.d_moms[4])

        # we compute alpha (initial guess)
        #alpha[0], alpha[1] = locRho/((np.pi*locT)**1.5), 1./locT
        # the missing factor in d_moms[4] is added here
        self.equiDistInitKern.prepared_call((1,1), (1,1,1),
            *self.ptr(self.d_moms))

        # compute the equilibrium BGK distribution
        self.equiDistComputeKern.prepared_call(self.grid, self.block,
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_fe.ptr, *self.ptr(self.d_moms)
        )

        # outKern
        self.outKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, 
            self.d_moms[0].ptr, self.d_moms[4].ptr, 
            self.d_fe.ptr, self.d_floc.ptr, d_Q)




"""
ESBGK "Iteration free" direct approach
The argument is: If the velocity grid is isotropic, and large enough; 
the error in conservation would be spectrally low
"""
class DGFSESBGKDirectGLLScatteringModel(DGFSScatteringModel):
    scattering_model = 'esbgk-direct-gll'

    def __init__(self, backend, cfg, velocitymesh):
        super().__init__(backend, cfg, velocitymesh)

    def load_parameters(self):
        Pr = self.cfg.getfloat('scattering-model', 'Pr', 2./3.)
        omega = self.cfg.getfloat('scattering-model', 'omega');
        muRef = self.cfg.getfloat('scattering-model', 'muRef');
        Tref = self.cfg.getfloat('scattering-model', 'Tref');

        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press

        self._prefactor = (t0*Pr*p0/visc)
        self._omega = omega
        self._Pr = Pr
        print("Pr:", self._Pr)
        print("prefactor:", self._prefactor)

    def perform_precomputation(self):
        # Precompute aa, bb1, bb2 (required for kernel)
        # compute l
        Nv = self.vm.Nv()
        Nrho = self.vm.Nrho()
        M = self.vm.M()
        L = self.vm.L()
        qz = self.vm.qz()
        qw = self.vm.qw()
        sz = self.vm.sz()
        vsize = self.vm.vsize()
        cv = self.vm.cv()
        self.cw = self.vm.cw()

        # the precision
        dtype = np.float64
        d = 'd'

        # number of alpha variables to be determined 
        self.nalph = 5
        nalph = self.nalph

        # number of variables for ESBGK
        self.nalphES = 10
        nalphES = self.nalphES

        # allocate velocity mesh in PyCUDA gpuarray
        self.d_cvx = gpuarray.to_gpu(cv[0,:])
        self.d_cvy = gpuarray.to_gpu(cv[1,:])
        self.d_cvz = gpuarray.to_gpu(cv[2,:])
        self.d_cSqr = gpuarray.to_gpu(np.einsum('ij,ij->j', cv, cv))

        # define scratch spaces
        # cell local distribution function
        self.d_floc = gpuarray.empty(vsize, dtype=dtype)

        # cell local equilibrium distribution function
        self.d_fe = gpuarray.empty_like(self.d_floc)         

        # cell local equilibrium distribution function (ESBGK)
        self.d_feES = gpuarray.empty_like(self.d_floc)         

        # scratch variable for storage for the moments
        self.d_mom10 = gpuarray.empty_like(self.d_floc)
        self.d_mom11 = gpuarray.empty_like(self.d_floc)
        self.d_mom12 = gpuarray.empty_like(self.d_floc)
        self.d_mom2 = gpuarray.empty_like(self.d_floc)

        # additional variables for ESBGK
        self.d_mom2es_xx = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_yy = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_zz = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_xy = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_yz = gpuarray.empty_like(self.d_floc)
        self.d_mom2es_zx = gpuarray.empty_like(self.d_floc)

        # storage for reduced moments 
        self.d_moms = [gpuarray.empty(1,dtype=dtype) for i in range(nalph)]

        # storage for reduced moments for ESBGK
        self.d_momsES = [gpuarray.empty(1,dtype=dtype) for i in range(nalphES)]
        self.d_equiMomsES = [
            gpuarray.empty(1,dtype=dtype) for i in range(nalphES)]

        # block size for running the kernel
        self.block = (256, 1, 1)
        self.grid = get_grid_for_block(self.block, vsize)

        # storage for alpha (ESBGK)
        self.d_alphaES = gpuarray.empty(nalphES, dtype=dtype)

        # for defining ptr
        self.ptr = lambda x: list(map(lambda v: v.ptr, x))

        # extract the template
        dfltargs = dict(cw=self.cw,
            vsize=vsize, prefac=self._prefactor, 
            soasz=self.backend.soasz, nalph=self.nalph, 
            omega=self._omega, Pr=self._Pr,
            block_size=self.block[0], nalphES=self.nalphES)
        src = DottedTemplateLookup(
            'frfs.solvers.dgfs.kernels.scattering', dfltargs
        ).get_template('esbgk-direct-gll').render()

        # Compile the source code and retrieve the kernel
        module = compiler.SourceModule(src)

        # sum kernel
        self.sumKern = module.get_function("sum_")
        self.sumKern.prepare('PPII')
        self.sumKern.set_cache_config(cuda.func_cache.PREFER_L1)
        grid = self.grid
        block = self.block
        seq_count0 = int(4)
        N = int(vsize)
        #grid_count0 = int((grid[0] + (-grid[0] % seq_count0)) // seq_count0)
        grid_count0 = int((grid[0]//seq_count0 + ceil(grid[0]%seq_count0)))
        d_st = gpuarray.empty(grid_count0, dtype=dtype)
        #seq_count1 = int((grid_count0 + (-grid_count0 % block[0])) // block[0])
        seq_count1 = int((grid_count0//block[0] + ceil(grid_count0%block[0])))
        def sum_(d_in, d_out):
            self.sumKern.prepared_call(
                (grid_count0,1), block, d_in.ptr, d_st.ptr, seq_count0, N)
            self.sumKern.prepared_call(
                (1,1), block, d_st.ptr, d_out.ptr, seq_count1, grid_count0)
        self.sumFunc = sum_

        # extract the element-local space coefficients
        self.flocKern = module.get_function("flocKern")
        self.flocKern.prepare('IIIIIIPP')
        self.flocKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # compute the first moment of local distribution
        self.mom1Kern = module.get_function("mom1")
        self.mom1Kern.prepare('PPP'+'P'+'PPP')
        self.mom1Kern.set_cache_config(cuda.func_cache.PREFER_L1)

        # helper function for recovering density/velocity from moments
        self.mom01NormKern = module.get_function("mom01Norm")
        self.mom01NormKern.prepare('PPPP')
        self.mom01NormKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # compute the second moment of local distribution
        self.mom2Kern = module.get_function("mom2")
        self.mom2Kern.prepare('PPP'+'PP'+'P'*4)
        self.mom2Kern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the equiDistInit kernel for execution
        self.equiDistInitKern = module.get_function("equiDistInit")
        self.equiDistInitKern.prepare('P'*nalph)
        self.equiDistInitKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the equiDistCompute kernel for execution
        self.equiDistComputeKern = module.get_function("equiDistCompute")
        self.equiDistComputeKern.prepare('PPPP'+'P'*nalph)
        self.equiDistComputeKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # compute the second moments for ESBGK
        self.mom2ESKern = module.get_function("mom2ES")
        self.mom2ESKern.prepare('PPP'+'PP'+'P'*4+'P'*6)
        self.mom2ESKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # compute the second moments for ESBGK
        self.mom2ESNormKern = module.get_function("mom2ESNorm")
        self.mom2ESNormKern.prepare('P'*nalph+'P'*nalphES)
        self.mom2ESNormKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # computes the equilibrium BGK distribution, and constructs expM
        self.equiESDistComputeKern = module.get_function("equiESDistCompute")
        self.equiESDistComputeKern.prepare('PPP'+'P'+'P'*nalphES)
        self.equiESDistComputeKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the out kernel for execution
        self.outKern = module.get_function("output")
        self.outKern.prepare('IIIIII'+'P'*2+'PPP')
        self.outKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # required by the child class (may be deleted by the child)
        self.module = module


    def fs(self, d_arr_in, d_arr_out, elem, upt):
        nrow, ldim, _ = d_arr_in.traits
        ncola, ncolb = d_arr_in.ioshape[1:]

        d_f0 = d_arr_in._as_parameter_
        d_Q = d_arr_out._as_parameter_

        # construct d_floc from d_f0
        self.flocKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, d_f0, self.d_floc.ptr)

        # compute first moment
        self.mom1Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, self.d_floc.ptr,
            self.d_mom10.ptr, self.d_mom11.ptr, self.d_mom12.ptr)

        # compute macroscopic properties
        self.sumFunc(self.d_floc, self.d_moms[0])  # missing: ${cw}
        self.sumFunc(self.d_mom10, self.d_moms[1]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom11, self.d_moms[2]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom12, self.d_moms[3]) # missing: ${cw}/d_moms[0]

        # insert the missing factor for the density and velocity
        self.mom01NormKern.prepared_call((1,1), (1,1,1), 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the second moments
        self.mom2Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_floc.ptr, self.d_mom2.ptr, 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr)

        # compute the temperature: missing factor cw/(1.5*d_moms[0])
        self.sumFunc(self.d_mom2, self.d_moms[4])

        # we compute alpha (initial guess)
        #alpha[0], alpha[1] = locRho/((np.pi*locT)**1.5), 1./locT
        # the missing factor in d_moms[4] is added here
        self.equiDistInitKern.prepared_call((1,1), (1,1,1),
            *self.ptr(self.d_moms))

        # compute the equilibrium BGK distribution
        self.equiDistComputeKern.prepared_call(self.grid, self.block,
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_fe.ptr, *self.ptr(self.d_moms)
        )

        # Now the ESBGK relaxation

        # compute the second moments for ESBGK
        self.mom2ESKern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_floc.ptr, 
            self.d_fe.ptr, 
            self.d_moms[0].ptr, self.d_moms[1].ptr, 
            self.d_moms[2].ptr, self.d_moms[3].ptr, 
            self.d_mom2es_xx.ptr, self.d_mom2es_yy.ptr, 
            self.d_mom2es_zz.ptr, self.d_mom2es_xy.ptr, 
            self.d_mom2es_yz.ptr, self.d_mom2es_zx.ptr
        )

        # reduce the moments 
        self.sumFunc(self.d_mom2es_xx, self.d_momsES[4])  # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_yy, self.d_momsES[5]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_zz, self.d_momsES[6]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_xy, self.d_momsES[7]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_yz, self.d_momsES[8]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom2es_zx, self.d_momsES[9]) # missing: ${cw}/d_moms[0]

        # normalize the moments (incorporates the missing factor)
        # (Also transfers first four entries of d_moms to d_momsES)
        # initializes alphaES
        self.mom2ESNormKern.prepared_call((1,1), (1,1,1), 
            *self.ptr(list(self.d_moms + self.d_momsES))
        )

        # compute the ESBGK distribution
        self.equiESDistComputeKern.prepared_call(self.grid, self.block,
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_feES.ptr, *self.ptr(self.d_momsES)
        )

        # outKern
        self.outKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, 
            self.d_moms[0].ptr, self.d_moms[4].ptr, 
            self.d_feES.ptr, self.d_floc.ptr, d_Q)



"""
Shakov "Iteration free" direct approach
The argument is: If the velocity grid is isotropic, and large enough; 
the error in conservation would be spectrally low
"""
class DGFSShakovDirectGLLScatteringModel(DGFSScatteringModel):
    scattering_model = 'shakov-direct-gll'

    def __init__(self, backend, cfg, velocitymesh):
        super().__init__(backend, cfg, velocitymesh)

    def load_parameters(self):
        Pr = self.cfg.getfloat('scattering-model', 'Pr', 2./3.)
        omega = self.cfg.getfloat('scattering-model', 'omega');
        muRef = self.cfg.getfloat('scattering-model', 'muRef');
        Tref = self.cfg.getfloat('scattering-model', 'Tref');

        t0 = self.vm.H0()/self.vm.u0() # non-dimensionalization time scale
        visc = muRef*((self.vm.T0()/Tref)**omega) # the viscosity
        p0 = self.vm.n0()*self.vm.R0/self.vm.NA*self.vm.T0() # non-dim press

        self._prefactor = (t0*p0/visc)
        #self._prefactor = (t0*Pr*p0/visc)
        self._omega = omega
        self._Pr = Pr
        print("Pr:", self._Pr)
        print("prefactor:", self._prefactor)

    def perform_precomputation(self):
        # Precompute aa, bb1, bb2 (required for kernel)
        # compute l
        Nv = self.vm.Nv()
        Nrho = self.vm.Nrho()
        M = self.vm.M()
        L = self.vm.L()
        qz = self.vm.qz()
        qw = self.vm.qw()
        sz = self.vm.sz()
        vsize = self.vm.vsize()
        cv = self.vm.cv()
        self.cw = self.vm.cw()

        # the precision
        dtype = np.float64
        d = 'd'

        # number of variables for ESBGK
        self.nalphSk = 8
        nalphSk = self.nalphSk

        # allocate velocity mesh in PyCUDA gpuarray
        self.d_cvx = gpuarray.to_gpu(cv[0,:])
        self.d_cvy = gpuarray.to_gpu(cv[1,:])
        self.d_cvz = gpuarray.to_gpu(cv[2,:])

        # define scratch spaces
        # cell local distribution function
        self.d_floc = gpuarray.empty(vsize, dtype=dtype)

        # cell local equilibrium distribution function (Shakov)
        self.d_feSk = gpuarray.empty_like(self.d_floc)         

        # scratch variable for storage for the moments
        self.d_mom10 = gpuarray.empty_like(self.d_floc)
        self.d_mom11 = gpuarray.empty_like(self.d_floc)
        self.d_mom12 = gpuarray.empty_like(self.d_floc)
        self.d_mom2 = gpuarray.empty_like(self.d_floc)

        # additional variables for ESBGK
        self.d_mom3sk_x = gpuarray.empty_like(self.d_floc)
        self.d_mom3sk_y = gpuarray.empty_like(self.d_floc)
        self.d_mom3sk_z = gpuarray.empty_like(self.d_floc)

        # storage for reduced moments for ESBGK
        self.d_momsSk = [gpuarray.empty(1,dtype=dtype) for i in range(nalphSk)]
        self.d_equiMomsSk = [
            gpuarray.empty(1,dtype=dtype) for i in range(nalphSk)]

        # block size for running the kernel
        self.block = (256, 1, 1)
        self.grid = get_grid_for_block(self.block, vsize)

        # storage for alpha (ESBGK)
        self.d_alphaSk = gpuarray.empty(nalphSk, dtype=dtype)

        # for defining ptr
        self.ptr = lambda x: list(map(lambda v: v.ptr, x))

        # extract the template
        dfltargs = dict(cw=self.cw,
            vsize=vsize, prefac=self._prefactor, 
            soasz=self.backend.soasz, 
            omega=self._omega, Pr=self._Pr,
            block_size=self.block[0], nalphSk=self.nalphSk)
        src = DottedTemplateLookup(
            'frfs.solvers.dgfs.kernels.scattering', dfltargs
        ).get_template('shakov-direct-gll').render()

        # Compile the source code and retrieve the kernel
        module = compiler.SourceModule(src)

        # sum kernel
        self.sumKern = module.get_function("sum_")
        self.sumKern.prepare('PPII')
        self.sumKern.set_cache_config(cuda.func_cache.PREFER_L1)
        grid = self.grid
        block = self.block
        seq_count0 = int(4)
        N = int(vsize)
        #grid_count0 = int((grid[0] + (-grid[0] % seq_count0)) // seq_count0)
        grid_count0 = int((grid[0]//seq_count0 + ceil(grid[0]%seq_count0)))
        d_st = gpuarray.empty(grid_count0, dtype=dtype)
        #seq_count1 = int((grid_count0 + (-grid_count0 % block[0])) // block[0])
        seq_count1 = int((grid_count0//block[0] + ceil(grid_count0%block[0])))
        def sum_(d_in, d_out):
            self.sumKern.prepared_call(
                (grid_count0,1), block, d_in.ptr, d_st.ptr, seq_count0, N)
            self.sumKern.prepared_call(
                (1,1), block, d_st.ptr, d_out.ptr, seq_count1, grid_count0)
        self.sumFunc = sum_

        # extract the element-local space coefficients
        self.flocKern = module.get_function("flocKern")
        self.flocKern.prepare('IIIIIIPP')
        self.flocKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # compute the first moment of local distribution
        self.mom1Kern = module.get_function("mom1")
        self.mom1Kern.prepare('PPP'+'P'+'PPP')
        self.mom1Kern.set_cache_config(cuda.func_cache.PREFER_L1)

        # helper function for recovering density/velocity from moments
        self.mom01NormKern = module.get_function("mom01Norm")
        self.mom01NormKern.prepare('PPPP')
        self.mom01NormKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # compute the second moment of local distribution
        self.mom23SkKern = module.get_function("mom23Sk")
        self.mom23SkKern.prepare('PPP'+'P'+'P'*4+'P'*4)
        self.mom23SkKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # compute the second moment of local distribution
        self.mom23SkNormKern = module.get_function("mom23SkNorm")
        self.mom23SkNormKern.prepare('P'*nalphSk)
        self.mom23SkNormKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # computes the equilibrium BGK distribution, and constructs expM
        self.equiSkDistComputeKern = module.get_function("equiSkDistCompute")
        self.equiSkDistComputeKern.prepare('PPP'+'P'+'P'*nalphSk)
        self.equiSkDistComputeKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # Prepare the out kernel for execution
        self.outKern = module.get_function("output")
        self.outKern.prepare('IIIIII'+'P'*2+'PPP')
        self.outKern.set_cache_config(cuda.func_cache.PREFER_L1)

        # required by the child class (may be deleted by the child)
        self.module = module


    def fs(self, d_arr_in, d_arr_out, elem, upt):
        nrow, ldim, _ = d_arr_in.traits
        ncola, ncolb = d_arr_in.ioshape[1:]

        d_f0 = d_arr_in._as_parameter_
        d_Q = d_arr_out._as_parameter_

        # construct d_floc from d_f0
        self.flocKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, d_f0, self.d_floc.ptr)

        # compute first moment
        self.mom1Kern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr, self.d_floc.ptr,
            self.d_mom10.ptr, self.d_mom11.ptr, self.d_mom12.ptr)

        # compute macroscopic properties
        self.sumFunc(self.d_floc, self.d_momsSk[0])  # missing: ${cw}
        self.sumFunc(self.d_mom10, self.d_momsSk[1]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom11, self.d_momsSk[2]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom12, self.d_momsSk[3]) # missing: ${cw}/d_moms[0]

        # insert the missing factor for the density and velocity
        self.mom01NormKern.prepared_call((1,1), (1,1,1), 
            self.d_momsSk[0].ptr, self.d_momsSk[1].ptr, 
            self.d_momsSk[2].ptr, self.d_momsSk[3].ptr)

        # compute the second and third moments for Shakov
        self.mom23SkKern.prepared_call(self.grid, self.block, 
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_floc.ptr, 
            self.d_momsSk[0].ptr, self.d_momsSk[1].ptr, 
            self.d_momsSk[2].ptr, self.d_momsSk[3].ptr, 
            self.d_mom2.ptr,
            self.d_mom3sk_x.ptr, self.d_mom3sk_y.ptr, 
            self.d_mom3sk_z.ptr
        )

        # compute the temperature: missing factor cw/(1.5*d_moms[0])
        self.sumFunc(self.d_mom2, self.d_momsSk[4])
        self.sumFunc(self.d_mom3sk_x, self.d_momsSk[5])  # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom3sk_y, self.d_momsSk[6]) # missing: ${cw}/d_moms[0]
        self.sumFunc(self.d_mom3sk_z, self.d_momsSk[7]) # missing: ${cw}/d_moms[0]

        # normalize the moments (incorporates the missing factor)
        # (Also transfers first four entries of d_moms to d_momsES)
        # initializes alphaES
        self.mom23SkNormKern.prepared_call((1,1), (1,1,1), 
            *self.ptr(self.d_momsSk)
        )

        # compute the ESBGK distribution
        self.equiSkDistComputeKern.prepared_call(self.grid, self.block,
            self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            self.d_feSk.ptr, *self.ptr(self.d_momsSk)
        )

        # outKern
        self.outKern.prepared_call(self.grid, self.block, 
            nrow, ldim, ncola, ncolb, elem, upt, 
            self.d_momsSk[0].ptr, self.d_momsSk[4].ptr, 
            self.d_feSk.ptr, self.d_floc.ptr, d_Q)
