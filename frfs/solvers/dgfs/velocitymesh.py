from frfs.quadratures import zwgj, zwglj
import numpy as np

from frfs.sphericaldesign import get_sphquadrule
from pycuda import gpuarray

class DGFSVelocityMesh(object):
    R0 = 8.3144598
    NA = 6.0221409e+23

    def __init__(self, backend, cfg):
        self.backend = backend
        self.cfg = cfg

        # Construct the velocity mesh
        self._construct_velocity_mesh()

        # Load the quadrature points (for integration) 
        self._load_quadrature()

        # Construct the spherical mesh
        self._construct_spherical_mesh()

    def _construct_velocity_mesh(self):
        # read the non-dimensional variables (length, temperature, density)
        self._H0 = self.cfg.getfloat('non-dim', 'H0') # meters
        self._T0 = self.cfg.getfloat('non-dim', 'T0') # K
        self._rho0 = self.cfg.getfloat('non-dim', 'rho0') # kg/m^3
        self._molarMass0 = self.cfg.getfloat('non-dim', 'molarMass0') # kg/mol
        self._n0 = self._rho0/self._molarMass0*self.NA
        self._u0 = np.sqrt(2*self.R0/self._molarMass0*self._T0)

        # define the velocity mesh
        self._Nv = self.cfg.getint('constants', 'Nv')
        self._NvBatchSize = self.cfg.getint('constants', 'NvBatchSize')
        self._vsize = self._Nv**3
        assert (self._vsize >= self._NvBatchSize), "Should be less"
        self._NvBatches = int(self._vsize/self._NvBatchSize) + int(
            (1 if self._vsize%self._NvBatchSize else 0) )
        assert (self._NvBatches*self._NvBatchSize==self._vsize), "Should match"

        #print(self._NvBatchSize)
        #print(self._NvBatches)
        #print(self._vsize)
        #raise ValueError("")

        self._Nrho = self.cfg.getint('constants', 'Nrho', int(self._Nv/2))
        print("Nrho: ", self._Nrho)

        _cmax = self.cfg.getfloat('velocity-mesh', 'cmax')
        _Tmax = self.cfg.getfloat('velocity-mesh', 'Tmax')
        _dev = self.cfg.getfloat('velocity-mesh', 'dev')

        # normalize maximum bulk velocity
        _cmax /= self._u0

        # normalize maximum bulk temperature
        _Tmax /= self._T0

        # define the length of the velocity mesh
        self._L = _cmax + _dev*np.sqrt(_Tmax);
        self._S = self._L*2.0/(3.0+np.sqrt(2.0));
        self._R = 2*self._S;
        print("velocityMesh: (%s %s)"%(-self._L,self._L))
        print("n0, u0: (%s %s)"%(self._n0, self._u0))
        print(
            "%s Batches of Size %s"%(self._NvBatches, self._NvBatchSize)
        )

        # define the weight of the velocity mesh
        self._cw = (2.0*self._L/self._Nv)**3
        c0 = np.linspace(-self._L+self._L/self._Nv, 
            self._L-self._L/self._Nv, self._Nv)
        #self._cv = c0[np.mgrid[0:self._Nv, 0:self._Nv, 0:self._Nv]]
        #self._cv = self._cv.reshape((3,self._vsize))
        self._cv = np.zeros((3, self._vsize))
        for l in range(self._vsize):
            I = int(l/(self._Nv*self._Nv))
            J = int((l%(self._Nv*self._Nv))/self._Nv);
            K = int((l%(self._Nv*self._Nv))%self._Nv);
            self._cv[0,l] = c0[I];
            self._cv[1,l] = c0[J];
            self._cv[2,l] = c0[K];

        # TODO: Need to transfer to device
        self._d_cvx = self.backend.const_matrix(
            np.ascontiguousarray(self._cv[0,:]).reshape(1, self._vsize)
        )
        self._d_cvy = self.backend.const_matrix(
            np.ascontiguousarray(self._cv[1,:]).reshape(1, self._vsize)
        )
        self._d_cvz = self.backend.const_matrix(
            np.ascontiguousarray(self._cv[2,:]).reshape(1, self._vsize)
        )
        #print(self._cv[0,:])
        #_ax = self._d_cvx.get()
        #_ay = self._d_cvy.get()
        #_az = self._d_cvz.get()
        #for i in range(self._vsize):
        #    print(self._cv[0, i], self._cv[1, i], self._cv[2, i], 
        #        _ax[0,i], _ay[0,i], _az[0,i])
        #assert np.allclose(self._cv[0,:], self._d_cvx.get()), "Error cvx"
        #assert np.allclose(self._cv[1,:], self._d_cvy.get()), "Error cvy"
        #assert np.allclose(self._cv[2,:], self._d_cvz.get()), "Error cvz"        
        #self._d_cvx = gpuarray.to_gpu(np.ascontiguousarray(self._cv[0,:]))
        #self._d_cvy = gpuarray.to_gpu(np.ascontiguousarray(self._cv[1,:]))
        #self._d_cvz = gpuarray.to_gpu(np.ascontiguousarray(self._cv[2,:]))
        #self._d_cv = self.backend.const_matrix(self._cv)
        #assert np.allclose(self._cv, self._d_cv.get()), "Error"

    def _load_quadrature(self):
        quadrule = self.cfg.get('velocity-mesh', 'quad-rule', 'jacobi')
        vquads = {
            'jacobi': zwgj,
            'lobatto': zwglj
        };
        a = self.cfg.getfloat('velocity-mesh', 'quad-rule-alpha', 0.0)
        b = self.cfg.getfloat('velocity-mesh', 'quad-rule-beta', 0.0)
        assert quadrule in vquads, "Valid quads:"+str(vquads.keys())

        # the default quadrules does not provide enough points
        self._qz, self._qw = vquads[quadrule](self._Nrho, a, b)  
        # scale the quadrature from [-1, 1] to [0, R] 
        self._qz = (self._R/2.0)*(1.0+self._qz)
        self._qw = ((self._R-0.0)/2.0)*self._qw

    def _construct_spherical_mesh(self):
        self._ssrule = self.cfg.get('spherical-design-rule', 'ssrule')
        self._M = self.cfg.getint('spherical-design-rule', 'M')
        srule = get_sphquadrule('symmetric', rule=self._ssrule,
            npts=2*self._M)
        # half-sphere
        self._sz = srule.pts[0:self._M, :]
        self._sw = 2*np.pi/self._M

    def Nv(self): return self._Nv
    def Nrho(self): return self._Nrho
    def NvBatches(self): return self._NvBatches
    def NvBatchSize(self): return self._NvBatchSize
    def M(self): return self._M
    def L(self): return self._L
    def vsize(self): return self._vsize
    def qz(self): return self._qz
    def qw(self): return self._qw
    def cv(self): return self._cv
    def cw(self): return self._cw
    def sz(self): return self._sz
    def sw(self): return self._sw
    def H0(self): return self._H0
    def T0(self): return self._T0
    def rho0(self): return self._rho0
    def n0(self): return self._n0
    def u0(self): return self._u0
    def molarMass0(self): return self._molarMass0
    
    #def d_cv(self): return self._d_cv
    def d_cvx(self): return self._d_cvx
    def d_cvy(self): return self._d_cvy
    def d_cvz(self): return self._d_cvz
    

