from abc import ABCMeta, abstractmethod
import numpy as np
from math import gamma
from frfs.util import ndrange
from frfs.nputil import npeval

class DGFSBiInitCondition(object, metaclass=ABCMeta):
    def __init__(self, backend, cfg, velocitymesh, name, **kwargs):
        self.backend = backend
        self.cfg = cfg
        self.vm = velocitymesh
        self._nspcs = self.vm.nspcs()
        self._init_vals = None

        print('\nReading initial condition for', name)
        # read model parameters
        self.load_parameters(name, **kwargs)

        # perform any necessary computation
        self.perform_precomputation()
        print('Finished initial condition precomputation for',name,'\n')

    @abstractmethod
    def load_parameters(self, name, **kwargs):
        pass

    @abstractmethod
    def perform_precomputation(self):
        pass

    def get_init_vals(self):
        return self._init_vals

    #def apply_init_vals(self, spcs_scal_upts_full, ele):
    #    assert len(spcs_scal_upts_full)==self._nspcs, "Should be in range"
    #    for ival, scal_upt in zip(self._init_vals, spcs_scal_upts_full):
    #        for j in range(self.vm.vsize()):
    #            scal_upt[:, j, :] = ival[j]

    def apply_init_vals(self, p, scal_upts_full, ele):
        assert p>=0 and p<self._nspcs, "Should be in range"
        initvals = self._init_vals[p]
        for j in range(self.vm.vsize()):
            scal_upts_full[:, j, :] = initvals[j]


class DGFSBiMaxwellianInitCondition(DGFSBiInitCondition):
    model = 'maxwellian'

    def __init__(self, backend, cfg, velocitymesh, name, **kwargs):
        super().__init__(backend, cfg, velocitymesh, name, **kwargs)

    def load_parameters(self, name, wall=False):
        u0 = self.vm.u0()
        self._ndenini = [1]*self._nspcs
        if not wall:
            for i in range(self._nspcs):
                prop = 'nden'+str(i+1)
                self._ndenini[i] = self.cfg.getfloat(name, prop)/self.vm.n0()
        self._Tini = self.cfg.getfloat(name, 'T')/self.vm.T0()
        uinix = self.cfg.getfloat(name, 'ux')/u0
        uiniy = self.cfg.getfloat(name, 'uy')/u0
        uiniz = self.cfg.getfloat(name, 'uz')/u0
        # because the cv is of shape (3, Nv)
        self._uini = np.array([uinix, uiniy, uiniz]).reshape((3,1))

    def perform_precomputation(self):
        m = self.vm.masses()
        self._init_vals = [0]*len(m)
        for p in range(self._nspcs):
            self._init_vals[p] = (
                self._ndenini[p]*((m[p]/np.pi/self._Tini)**1.5)*np.exp(-m[p]
                    *np.sum((self.vm.cv()-self._uini)**2, axis=0)/self._Tini)
            )

        # test the distribution support
        cv = self.vm.cv()
        vsize = self.vm.vsize()
        cw = self.vm.cw()
        T0 = self.vm.T0()
        n0 = self.vm.n0()
        molarMass0 = self.vm.molarMass0()
        u0 = self.vm.u0()

        for p in range(self._nspcs):
            mr = m[p]
            mcw = mr*cw

            ele_sol = np.zeros(5)
            soln = self._init_vals[p]

            #non-dimensional mass density
            ele_sol[0] = np.sum(soln)*mcw

            #non-dimensional velocities
            ele_sol[1] = np.tensordot(soln, cv[0,:], axes=(0,0))*mcw
            ele_sol[1] /= ele_sol[0]
            ele_sol[2] = np.tensordot(soln, cv[1,:], axes=(0,0))*mcw
            ele_sol[2] /= ele_sol[0]
            ele_sol[3] = np.tensordot(soln, cv[2,:], axes=(0,0))*mcw
            ele_sol[3] /= ele_sol[0]

            # peculiar velocity
            cx = cv[0,:]-ele_sol[1]
            cy = cv[1,:]-ele_sol[2]
            cz = cv[2,:]-ele_sol[3]
            cSqr = cx*cx + cy*cy + cz*cz

            # non-dimensional temperature
            ele_sol[4] = np.sum(soln*cSqr, axis=0)*(2.0/3.0*mcw*mr)
            ele_sol[4] /= ele_sol[0]

            print("for species:", p)
            print("bulk-property: input calculated")
            print("number-density:", self._ndenini[p]*n0, ele_sol[0]*n0/mr)
            print("x-velocity:", self._uini[0,0]*u0, ele_sol[1]*u0)
            print("y-velocity:", self._uini[1,0]*u0, ele_sol[2]*u0)
            print("z-velocity:", self._uini[2,0]*u0, ele_sol[3]*u0)
            print("temperature:", self._Tini*T0, ele_sol[4]*T0)

            if( not(
                np.allclose(self._ndenini[p]*n0, ele_sol[0]*n0/mr, atol=1e-5)
                and
                np.allclose(self._uini[0,0]*u0, ele_sol[1]*u0, atol=1e-5)
                and
                np.allclose(self._uini[1,0]*u0, ele_sol[2]*u0, atol=1e-5)
                and
                np.allclose(self._uini[2,0]*u0, ele_sol[3]*u0, atol=1e-5)
                and
                np.allclose(self._Tini*T0, ele_sol[4]*T0, atol=1e-5)
            )):
                raise ValueError("Bulk properties not conserved! See Nv,dev")

    def unondim(self):
        return self._uini


class DGFSBiMaxwellianExprInitCondition(DGFSBiInitCondition):
    model = 'maxwellianExpr'

    def __init__(self, backend, cfg, velocitymesh, name, **kwargs):
        super().__init__(backend, cfg, velocitymesh, name, **kwargs)

    def load_parameters(self, name):
        self._vars = self.cfg.items_as('constants', float)
        self.name = name
        if any(d in self._vars for d in 'xyz'):
            raise ValueError('Invalid constants (x, y, or z) in config file')

    def perform_precomputation(self):
        pass

    def maxwellian(self, m, ndenini, uxini, uyini, uzini, Tini):
        uini = np.array([uxini, uyini, uzini]).reshape((3,1))
        soln = ndenini*(((m/np.pi/Tini)**1.5)*
                np.exp(-m*np.sum((self.vm.cv()-uini)**2, axis=0)/Tini)
        )

        # test the distribution support
        nden_bulk = np.sum(soln)*self.vm.cw()
        if( not(
            np.allclose(ndenini, nden_bulk, atol=1e-5)
        )):
            raise ValueError("Bulk properties not conserved! Check Nv, dev")

        return soln

    def unondim(self):
        raise ValueError("this model depends on spatial coordinates!")

    def get_init_vals(self):
        raise ValueError("this model depends on spatial coordinates!")

    def apply_init_vals(self, p, scal_upts_full, ele):
        assert p>=0 and p<self._nspcs, "Should be in range"
        m = self.vm.masses()[p]

        vars = self._vars

        # Get the physical location of each solution point
        coords = ele.ploc_at_np('upts').swapaxes(0, 1)
        vars.update(dict(zip('xyz', coords)))

        # Evaluate the ICs from the config file
        ndenini = npeval(self.cfg.getexpr(self.name,'nden'+str(p+1)),vars)
        ndenini /= self.vm.n0()
        uxini = npeval(self.cfg.getexpr(self.name,'ux'), vars)/self.vm.u0()
        uyini = npeval(self.cfg.getexpr(self.name,'uy'), vars)/self.vm.u0()
        uzini = npeval(self.cfg.getexpr(self.name,'uz'), vars)/self.vm.u0()
        Tini = npeval(self.cfg.getexpr(self.name,'T'), vars)/self.vm.T0()

        def isf(data): return isinstance(data, float)

        if all(map(isf, [ndenini, uxini, uyini, uzini, Tini])):
            mxwl = self.maxwellian(m, ndenini,uxini, uyini, uzini,Tini)
            for j in range(self.vm.vsize()):
                scal_upts_full[:, j, :] = mxwl[j]
        else:
            def shape(ds): 
                if isf(ds): return np.full((ele.nupts, ele.neles), ds)
                else: return ds

            ndenini, uxini, uyini, uzini,Tini = map(shape, 
                [ndenini, uxini, uyini, uzini,Tini])

            for upt, elem in ndrange(ele.nupts, ele.neles):
                scal_upts_full[upt, :, elem] = self.maxwellian(
                    m, ndenini[upt, elem],
                    uxini[upt, elem], uyini[upt, elem], uzini[upt, elem],
                    Tini[upt, elem]
                )

