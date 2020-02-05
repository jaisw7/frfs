# -*- coding: utf-8 -*-

from frfs.integrators.std.base import BaseStdIntegrator
from frfs.util import memoize, proxylist

class BaseStdStepper(BaseStdIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add kernel cache
        self._axnpby_kerns = {}

    def collect_stats(self, stats):
        super().collect_stats(stats)

        # Total number of RHS evaluations
        stats.set('solver-time-integrator', 'nfevals', self._stepper_nfevals)


class StdEulerDGFSStepper(BaseStdStepper):
    stepper_name = 'euler-dgfs'

    @property
    def _stepper_has_errest(self):
        return False

    @property
    def _stepper_nfevals(self):
        return self.nsteps

    @property
    def _stepper_nregs(self):
        return 2

    @property
    def _stepper_order(self):
        return 1

    def step(self, t, dt):
        add, rhs = self._add_dgfs, self.system.rhs
        collide, updatebc = self.system.collide, self.system.updatebc
        copyto, copyfrm = self._copy_to_reg_dgfs, self._copy_from_reg_dgfs
        invmass = self.system.invmass

        vm = self.system.vm
        NvBS = vm.NvBatchSize();

        # Get the bank indices for each register (n, n+1, rhs)
        r0, r1 = self._regidx

        # Ensure r0 references the bank containing u(t)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # collide: r1_full = Q(r0_full)
        collide(t, r0, r1) 

        # multiply by inverse mass
        #for batch in range(vm.NvBatches()):
        #    sidx, eidx = batch*NvBS, (batch+1)*NvBS
        #    copyto(r1, r1, sidx, eidx-1, 0, NvBS-1)            
        #    invmass(r1, r0)
        #    copyfrm(r0, r1, 0, NvBS-1, sidx, eidx-1)

        # Now add the advection part

        # update the boundary condition
        #if (self.nacptsteps==0):
        for batch in range(vm.NvBatches()):
            sidx, eidx = batch*NvBS, (batch+1)*NvBS

            # transfer chunk from r0_full to r0_reg
            copyto(r0, r0, sidx, eidx-1, 0, NvBS-1)

            # update the boundary coeffiecients based on soln stored in r0
            updatebc(t, r0, r1, batch, sidx)

        # We will call rhs multiple times based on NvBatchSize
        for batch in range(vm.NvBatches()):
            sidx, eidx = batch*NvBS, (batch+1)*NvBS

            # transfer chunk from r0_full to r0_reg
            copyto(r0, r0, sidx, eidx-1, 0, NvBS-1)

            # rhs: r1_reg = -∇·f(r0_reg);
            rhs(t, r0, r1, batch, sidx)

            # add the advection part: r0_reg = r0_reg + dt*r1_reg
            add(1.0, r0, dt, r1)

            # transfer chunk from r1_full to r1_reg
            copyto(r1, r1, sidx, eidx-1, 0, NvBS-1)

            # add the collision part: r0_reg = r0_reg + dt*r1_reg
            add(1.0, r0, dt, r1)

            # transfer chunk from r0_reg to r0_full for next step
            copyfrm(r0, r0, 0, NvBS-1, sidx, eidx-1)

        return r0


class StdTVDRK2DGFSStepper(BaseStdStepper):
    stepper_name = 'tvd-rk2-dgfs'

    @property
    def _stepper_has_errest(self):
        return False

    @property
    def _stepper_nfevals(self):
        return 2*self.nsteps

    @property
    def _stepper_nregs(self):
        return 3

    @property
    def _stepper_order(self):
        return 2

    def step(self, t, dt):
        add, rhs = self._add_dgfs, self.system.rhs
        collide, updatebc = self.system.collide, self.system.updatebc
        copyto, copyfrm = self._copy_to_reg_dgfs, self._copy_from_reg_dgfs
        invmass = self.system.invmass

        vm = self.system.vm

        # Get the bank indices for each register (n, n+1, rhs)
        r0, r1, r2 = self._regidx

        # Ensure r0 references the bank containing u(t)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage; r1 = -∇·f(r0); r0 = r0 + dt*r1
        # call collide
        collide(t, r0, r1) 
        
        # Now add the advection part
        NvBS = vm.NvBatchSize();

        # multiply by inverse mass
        #for batch in range(vm.NvBatches()):
        #    sidx, eidx = batch*NvBS, (batch+1)*NvBS
        #    copyto(r1, r1, sidx, eidx-1, 0, NvBS-1)            
        #    invmass(r1, r0)
        #    copyfrm(r0, r1, 0, NvBS-1, sidx, eidx-1)

        # update the boundary conditions
        for batch in range(vm.NvBatches()):
            sidx, eidx = batch*NvBS, (batch+1)*NvBS
            copyto(r0, r0, sidx, eidx-1, 0, NvBS-1)
            updatebc(t, r0, r1, batch, sidx)

        # We will call rhs multiple times based on NvBatchSize
        for batch in range(vm.NvBatches()):
            sidx = batch*NvBS
            eidx = (batch+1)*NvBS

            # transfer chunk from r0_full to r0_reg
            copyto(r0, r0, sidx, eidx-1, 0, NvBS-1)

            # call rhs 
            rhs(t, r0, r1, batch, sidx)

            # call add to add the advection part
            add(1.0, r0, dt, r1)

            # transfer chunk from r1_full to r1_reg
            copyto(r1, r1, sidx, eidx-1, 0, NvBS-1)

            # call add to add the collision part
            add(dt, r1, 1.0, r0)

            # transfer chunk from r1_reg to r1_full
            copyfrm(r1, r1, 0, NvBS-1, sidx, eidx-1)

        # Second stage; r2 = -∇·f(r1); r0 = 0.5*r0 + 0.5*r1 + 0.5*dt*r2
        # call collide
        collide(t+dt, r1, r2) 

        # multiply by inverse mass
        #for batch in range(vm.NvBatches()):
        #    sidx, eidx = batch*NvBS, (batch+1)*NvBS
        #    copyto(r2, r2, sidx, eidx-1, 0, NvBS-1)            
        #    invmass(r2, r1)
        #    copyfrm(r1, r2, 0, NvBS-1, sidx, eidx-1)

        # update the boundary conditions
        for batch in range(vm.NvBatches()):
            sidx, eidx = batch*NvBS, (batch+1)*NvBS
            copyto(r1, r1, sidx, eidx-1, 0, NvBS-1)
            updatebc(t, r1, r2, batch, sidx)

        # We will call rhs multiple times based on NvBatchSize
        for batch in range(vm.NvBatches()):
            sidx = batch*NvBS
            eidx = (batch+1)*NvBS

            # transfer chunk from r2_full to r2_reg
            copyto(r2, r2, sidx, eidx-1, 0, NvBS-1)

            # transfer chunk from r0_full to r0_reg
            copyto(r0, r0, sidx, eidx-1, 0, NvBS-1)

            # call add to add the collision part
            add(0.5, r0, 0.5*dt, r2)

            # transfer chunk from r1_full to r1_reg
            copyto(r1, r1, sidx, eidx-1, 0, NvBS-1)

            # call rhs 
            rhs(t+dt, r1, r2, batch, sidx)

            # call add to add the advection part and previous step soln
            add(1.0, r0, 0.5, r1, 0.5*dt, r2)

            # transfer chunk from r0_reg to r0_full
            copyfrm(r0, r0, 0, NvBS-1, sidx, eidx-1)


        # Return the index of the bank containing u(t + dt)
        return r0


# need to generalize this
class StdEulerDGFSBiStepper(BaseStdStepper):
    stepper_name = 'euler-dgfsbi'

    @property
    def _stepper_has_errest(self):
        return False

    @property
    def _stepper_nfevals(self):
        return self.nsteps

    @property
    def _stepper_nregs(self):
        # must be nspcs*_stepper_nregs_orig
        return 4

    @property 
    def _stepper_nregs_orig(self):
        # offset for the second distribution
        return 2

    @property
    def _stepper_order(self):
        return 1

    def step(self, t, dt):
        add, rhs = self._add_dgfs, self.system.rhs
        collide, updatebc = self.system.collide, self.system.updatebc
        copyto, copyfrm = self._copy_to_reg_dgfs, self._copy_from_reg_dgfs
        invmass = self.system.invmass

        vm = self.system.vm

        # r00 = f^{(1)}
        # r10 = f^{(2)}

        # Get the bank indices for each register (n, n+1, rhs)
        #r00, r01, r10, r11 = self._regidx

        # Ensure r00, r10 references the bank containing u(t)
        if 0 != self._idxcurr:
            # the assumption is that the zeroth references current soln
            raise ValueError("Something wrong")
            r00, r01 = r01, r00
            r10, r11 = r11, r10

        # call collide
        # r01 = Q_{11} + Q_{12}
        # r11 = Q_{21} + Q_{22}
        collide(t, list(self._regidx[::self._stepper_nregs_orig]
            +self._regidx[1::self._stepper_nregs_orig]))


        # Now add the advection part
        NvBS = vm.NvBatchSize();

        # multiply by inverse mass
        #for p in range(vm.nspcs()):
        #    r0 = self._stepper_nregs_orig*p
        #    for batch in range(vm.NvBatches()):
        #        sidx, eidx = batch*NvBS, (batch+1)*NvBS
        #        copyto(r0+1, r0+1, sidx, eidx-1, 0, NvBS-1)  
        #        invmass(r0+1, r0)
        #        copyfrm(r0, r0+1, 0, NvBS-1, sidx, eidx-1)

        for p in range(vm.nspcs()):
            r0 = self._stepper_nregs_orig*p
            for batch in range(vm.NvBatches()):
                sidx, eidx = batch*NvBS, (batch+1)*NvBS
                copyto(r0, r0, sidx, eidx-1, 0, NvBS-1)
                updatebc(t, str(p), r0, r0+1, batch, sidx)

        # We will call rhs multiple times based on NvBatchSize
        for p in range(vm.nspcs()):
            r0 = self._stepper_nregs_orig*p
            for batch in range(vm.NvBatches()):
                sidx, eidx = batch*NvBS, (batch+1)*NvBS

                # transfer chunk from r0_full to r0_reg
                copyto(r0, r0, sidx, eidx-1, 0, NvBS-1)

                # call rhs 
                rhs(t, str(p), r0, r0+1, batch, sidx)

                # call add to add the advection part
                add(1.0, r0, dt, r0+1)

                # transfer chunk from r1_full to r1_reg
                copyto(r0+1, r0+1, sidx, eidx-1, 0, NvBS-1)

                # call add to add the collision part
                add(1.0, r0, dt, r0+1)

                # transfer chunk from r0_reg to r0_full
                copyfrm(r0, r0, 0, NvBS-1, sidx, eidx-1)

        return 0


# need to generalize this
class StdTVDRK2DGFSBiStepper(BaseStdStepper):
    stepper_name = 'tvd-rk2-dgfsbi'

    @property
    def _stepper_has_errest(self):
        return False

    @property
    def _stepper_nfevals(self):
        return 2*self.nsteps

    @property
    def _stepper_nregs(self):
        # must be nspcs*_stepper_nregs_orig
        return 6

    @property 
    def _stepper_nregs_orig(self):
        # offset for the second distribution
        return 3

    @property
    def _stepper_order(self):
        return 2

    def step(self, t, dt):
        add, rhs = self._add_dgfs, self.system.rhs
        collide, updatebc = self.system.collide, self.system.updatebc
        copyto, copyfrm = self._copy_to_reg_dgfs, self._copy_from_reg_dgfs
        invmass = self.system.invmass

        vm = self.system.vm

        # r00 = f^{(1)}
        # r10 = f^{(2)}

        # Get the bank indices for each register (n, n+1, rhs)
        #r00, r01, r02, r10, r11, r12 = self._regidx

        # Ensure r00, r10 references the bank containing u(t)
        if 0 != self._idxcurr:
            # the assumption is that the zeroth references current soln
            raise ValueError("Something wrong")
            r00, r01 = r01, r00
            r10, r11 = r11, r10

        # call collide
        # r01 = Q_{11} + Q_{12}
        # r11 = Q_{21} + Q_{22}
        collide(t, list(self._regidx[::self._stepper_nregs_orig]
            +self._regidx[1::self._stepper_nregs_orig]))

        # Now add the advection part
        NvBS = vm.NvBatchSize();

        # multiply by inverse mass
        #for p in range(vm.nspcs()):
        #    r0 = self._stepper_nregs_orig*p
        #    for batch in range(vm.NvBatches()):
        #        sidx, eidx = batch*NvBS, (batch+1)*NvBS
        #        copyto(r0+1, r0+1, sidx, eidx-1, 0, NvBS-1)  
        #        invmass(r0+1, r0)
        #        copyfrm(r0, r0+1, 0, NvBS-1, sidx, eidx-1)

        for p in range(vm.nspcs()):
            r0 = self._stepper_nregs_orig*p
            for batch in range(vm.NvBatches()):
                sidx, eidx = batch*NvBS, (batch+1)*NvBS
                copyto(r0, r0, sidx, eidx-1, 0, NvBS-1)
                updatebc(t, str(p), r0, r0+1, batch, sidx)

        # We will call rhs multiple times based on NvBatchSize
        for p in range(vm.nspcs()):
            r0 = self._stepper_nregs_orig*p
            for batch in range(vm.NvBatches()):
                sidx, eidx = batch*NvBS, (batch+1)*NvBS

                # transfer chunk from r0_full to r0_reg
                copyto(r0, r0, sidx, eidx-1, 0, NvBS-1)

                # call rhs 
                rhs(t, str(p), r0, r0+1, batch, sidx)

                # call add to add the advection part
                add(1.0, r0, dt, r0+1)

                # transfer chunk from r1_full to r1_reg
                copyto(r0+1, r0+1, sidx, eidx-1, 0, NvBS-1)

                # call add to add the collision part
                add(dt, r0+1, 1.0, r0)

                # transfer chunk from r0_reg to r0_full
                copyfrm(r0+1, r0+1, 0, NvBS-1, sidx, eidx-1)


        # the new solution is in r1
        # call collide
        # r02 = Q_{11} + Q_{12}
        # r12 = Q_{21} + Q_{22}
        collide(t, list(self._regidx[1::self._stepper_nregs_orig]
            +self._regidx[2::self._stepper_nregs_orig]))

        # multiply by inverse mass
        #for p in range(vm.nspcs()):
        #    r0 = self._stepper_nregs_orig*p
        #    for batch in range(vm.NvBatches()):
        #        sidx, eidx = batch*NvBS, (batch+1)*NvBS
        #        copyto(r0+2, r0+2, sidx, eidx-1, 0, NvBS-1)  
        #        invmass(r0+2, r0+1)
        #        copyfrm(r0+1, r0+2, 0, NvBS-1, sidx, eidx-1)

        for p in range(vm.nspcs()):
            r0 = self._stepper_nregs_orig*p
            for batch in range(vm.NvBatches()):
                sidx, eidx = batch*NvBS, (batch+1)*NvBS
                copyto(r0+1, r0+1, sidx, eidx-1, 0, NvBS-1)
                updatebc(t, str(p), r0+1, r0+2, batch, sidx)

        # Now add the advection part
        # We will call rhs multiple times based on NvBatchSize
        for p in range(vm.nspcs()):
            r0 = self._stepper_nregs_orig*p
            for batch in range(vm.NvBatches()):
                sidx, eidx = batch*NvBS, (batch+1)*NvBS

                # transfer chunk from r2_full to r2_reg
                copyto(r0+2, r0+2, sidx, eidx-1, 0, NvBS-1)

                # transfer chunk from r0_full to r0_reg
                copyto(r0, r0, sidx, eidx-1, 0, NvBS-1)

                # call add to add the collision part
                add(0.5, r0, 0.5*dt, r0+2)

                # transfer chunk from r1_full to r1_reg
                copyto(r0+1, r0+1, sidx, eidx-1, 0, NvBS-1)

                # call rhs 
                rhs(t+dt, str(p), r0+1, r0+2, batch, sidx)

                # call add to add the advection part and previous step soln
                add(1.0, r0, 0.5, r0+1, 0.5*dt, r0+2)

                # transfer chunk from r0_reg to r0_full
                copyfrm(r0, r0, 0, NvBS-1, sidx, eidx-1)

        return 0









#--- Asymptotic schemes


class AstdStepper(BaseStdStepper):
    stepper_name = None

    @property
    def _stepper_has_errest(self):
        return False

    @property
    def _stepper_nfevals(self):
        pass


# """a-stage explicit, b-stage implicit, c-order ARS integration scheme"""
class ARSabcAstdStepper(AstdStepper):
    stepper_name = None

    @property
    def _stepper_nstages(self):
        pass

    @property
    def _stepper_nregs_moms(self):
        pass

    @property
    def _stepper_nregs(self): 
        # the first register is for soln, the last is for scratch
        pass

    def explicit(self, t, fr0, fr1, r0, r1):
        rhs, updatebc = self.system.rhs, self.system.updatebc
        vm = self.system.vm
        NvBatches, NvBS = vm.NvBatches(), vm.NvBatchSize()
        copyto, copyfrm = self._copy_to_reg_dgfs, self._copy_from_reg_dgfs

        # update the boundary condition
        for batch in range(NvBatches):
            sidx, eidx = batch*NvBS, (batch+1)*NvBS
            copyto(fr0, r0, sidx, eidx-1, 0, NvBS-1)
            updatebc(t, r0, fr1, batch, sidx)

        # We will call rhs multiple times based on NvBatchSize
        for batch in range(NvBatches):
            sidx, eidx = batch*NvBS, (batch+1)*NvBS
            copyto(fr0, r0, sidx, eidx-1, 0, NvBS-1) # r0 <-- fr0
            rhs(t, r0, r1, batch, sidx) # r1 = -∇·f(r0);
            copyfrm(r1, fr1, 0, NvBS-1, sidx, eidx-1) # fr1 <-- r1


    def step(self, t, dt):
        sys = self.system
        moment, updateMoment = sys.moment, sys.updateMomentARS
        updateDist, consMaxwellian = sys.updateDistARS, sys.constructMaxwellian

        l = self._stepper_order;

        # Get the bank indices for each register (n, n+1, rhs)
        scratch = self._regidx
        if scratch[0] != self._idxcurr: 
            scratch[0], scratch[1] = scratch[1], scratch[0]
        F, L, M = scratch[:-1:3], scratch[1:-1:3], scratch[2:-1:3]
        fn, frs = scratch[0], scratch[-1]  # last register is actually scratch

        # scratch registers
        r0, r1 = 0, 1
        if r0 != self._idxcurr: r0, r1 = r1, r0
        scratch_moms = sys.get_nregs_moms(self._stepper_nregs_moms)
        U, LU = scratch_moms[:-1:2], scratch_moms[1:-1:2]
        mrs = scratch_moms[-1] # last register is actually scratch

        # the first and the last registers are the initial data
        F = F + [fn]

        # Compute the moment of the initial distribution
        moment(t, fn, U[0], frs)

        # loop over the stages
        for i in range(self._stepper_nstages):

            _Aj, Aj = self._A[i+1][0:i+1], self.A[i+1][1:i+2]

            # Compute the explicit part; L[i] = -∇·f(d_ucoeff);
            self.explicit(t, F[i], L[i], r0, r1)

            # Compute the moment of the explicit part
            moment(t, L[i], LU[i], frs)

            # update the moments
            updateMoment(dt, *[*_Aj, *Aj, *LU[:i+1], *U[:i+2]])

            # implictly construct the Maxwellian/Gaussian given moments
            consMaxwellian(t, U[i+1], M[i], mrs)

            # update the distribution
            updateDist(dt, *[*_Aj, *Aj, *L[:i+1], *U[:i+2], 
                *M[:i+1], *F[:i+2]])

        return F[self._stepper_nstages]

        

# """1-stage explicit, 1-stage implicit, 1-order ARS integration scheme"""
class ARS111AstdStepper(ARSabcAstdStepper):
    stepper_name = "adgfs-ars-111"

    _A = [[0., 0.], [1., 0.]]
    A = [[0., 0.], [0., 1.]]

    @property
    def _stepper_nfevals(self): return 1*self.nsteps

    @property
    def _stepper_nregs(self): return 1+1+1+1

    @property
    def _stepper_nregs_moms(self): return 3+1

    @property
    def _stepper_order(self): return 1

    @property
    def _stepper_nstages(self): return 1


# """2-stage explicit, 2-stage implicit, 2-order ARS integration scheme"""
class ARS222AstdStepper(ARSabcAstdStepper):
    stepper_name = "adgfs-ars-222"

    gamma = 1.-(2.**0.5)/2.
    delta = 1. - 1./(2.*gamma)
    _A = [[0., 0., 0.], [gamma, 0., 0.], [delta, 1-delta, 0.]]
    A = [[0., 0., 0.], [0., gamma, 0.], [0., 1-gamma, gamma]]

    @property
    def _stepper_nfevals(self): return 2*self.nsteps

    @property
    def _stepper_nregs(self): return 2+2+2+1

    @property
    def _stepper_nregs_moms(self): return 5+1

    @property
    def _stepper_order(self): return 2

    @property
    def _stepper_nstages(self): return 2


# """4-stage explicit, 4-stage implicit, 3-order ARS integration scheme"""
class ARS443AstdStepper(ARSabcAstdStepper):
    stepper_name = "adgfs-ars-443"

    _A = [
        [0., 0., 0., 0., 0.], 
        [1./2., 0., 0., 0., 0.], 
        [11./18., 1./18., 0., 0., 0.],
        [5./6., -5./6., 1./2., 0., 0.],
        [1./4., 7./4., 3./4., -7./4., 0.]
    ]
    A = [
        [0., 0., 0., 0., 0.], 
        [0., 1./2., 0., 0., 0.], 
        [0., 1./6., 1./2., 0., 0.],
        [0., -1./2., 1./2., 1./2., 0.],
        [0., 3./2., -3./2., 1./2., 1./2.]
    ]

    @property
    def _stepper_nfevals(self): return 4*self.nsteps

    @property
    def _stepper_nregs(self): return 4+4+4+1

    @property
    def _stepper_nregs_moms(self): return 9+1

    @property
    def _stepper_order(self): return 3

    @property
    def _stepper_nstages(self): return 4



# """1st-order IMEX backward difference integration scheme"""
class BDF1AstdStepper(ARS111AstdStepper):
    stepper_name = 'adgfs-bdf-1'


# """2nd-order IMEX backward difference integration scheme"""
class BDF2AstdStepper(ARS222AstdStepper):
    stepper_name = "adgfs-bdf-2"

    bdf2_A = [1./3., -4./3., 1.]
    bdf2_G = [-2./3., 4./3.]
    bdf2_B = 2./3.
    nstepsInit = 10

    @property
    def _stepper_nfevals(self): return 1*self.nsteps

    @property
    def _stepper_order(self): return 2

    @property
    def _stepper_nstages(self): return 2

    def step(self, t, dt):
        sys = self.system
        copyd, copym = sys.copy_dist, sys.copy_moms
        moment, updateMoment = sys.moment, sys.updateMomentBDF
        updateDist, consMaxwellian = sys.updateDistBDF, sys.constructMaxwellian

        # Get the bank indices for each register (n, n+1, rhs)
        scratch = self._regidx
        if scratch[0] != self._idxcurr: 
            scratch[0], scratch[1] = scratch[1], scratch[0]
        L0, f0, L1, f1, M = scratch[1:-1]
        fn, frs = scratch[0], scratch[-1]

        # scratch registers
        r0, r1 = 0, 1
        if r0 != self._idxcurr: r0, r1 = r1, r0
        scratch_moms = sys.get_nregs_moms(self._stepper_nregs_moms)
        U0, LU0, U1, LU1, U, mrs = scratch_moms

        a1, a2, a3 = self.bdf2_A
        g1, g2  = self.bdf2_G
        b = self.bdf2_B

        # if this is the first step, use the ars-222 scheme
        if self.nacptsteps == 0:
            tloc, dtloc = t, dt/self.nstepsInit
            fno = []
            for T, _ in enumerate(zip(sys.ele_types)):
                fno.append(sys.eles_scal_upts_inb_full[T][fn].get())

            for step in range(self.nstepsInit):
                super().step(tloc, dtloc);
                tloc += dtloc

            for T, _ in enumerate(zip(sys.ele_types)):
                sys.eles_scal_upts_inb_full[T][f1].set(fno[T])

            moment(t, f1, U1, frs); self.explicit(t, f1, L1, r0, r1); 
            moment(t, L1, LU1, frs);
            del fno; return fn

        # Compute the moment of the initial distribution
        copym(U0, U1);
        moment(t, fn, U1, frs);

        # Compute the explicit part; L1 = -∇·f(fn);
        copyd(L0, L1);
        self.explicit(t, fn, L1, r0, r1)

        # Compute the moment of the explicit part
        copym(LU0, LU1); 
        moment(t, L1, LU1, frs)

        # update the moments
        updateMoment(dt, a1, U0, -g1, LU0, a2, U1, -g2, LU1, a3, U, b)

        # implictly construct the Maxwellian (or Gaussian, etc.) given moments
        consMaxwellian(t, U, M, mrs)

        # update the distribution
        copyd(f0, f1); copyd(f1, fn);
        updateDist(dt, a1, f0, -g1, L0, a2, f1, -g2, L1, b, M, a3, U, fn)

        return fn




# """2nd-order IMEX backward difference integration scheme"""
class BDF3AstdStepper(ARS222AstdStepper):
    stepper_name = "adgfs-bdf-3"

    bdf3_A = [-2./11., 9./11., -18./11., 1.]
    bdf3_G = [6./11., -18./11., 18./11.]
    bdf3_B = 6./11.
    nstepsInit = 100

    @property
    def _stepper_nfevals(self): return 1*self.nsteps

    @property
    def _stepper_order(self): return 3

    @property
    def _stepper_nregs(self): return 2+2+2+2+1

    @property
    def _stepper_nregs_moms(self): return 7+1

    def step(self, t, dt):
        sys = self.system
        copyd, copym = sys.copy_dist, sys.copy_moms
        moment, updateMoment = sys.moment, sys.updateMomentBDF
        updateDist, consMaxwellian = sys.updateDistBDF, sys.constructMaxwellian

        # Get the bank indices for each register (n, n+1, rhs)
        scratch = self._regidx
        L0, f0, L1, f1, L2, f2, M = scratch[1:-1]
        fn, frs = scratch[0], scratch[-1]

        # scratch registers
        r0, r1 = 0, 1
        if r0 != self._idxcurr: r0, r1 = r1, r0

        scratch_moms = sys.get_nregs_moms(self._stepper_nregs_moms)
        U0, LU0, U1, LU1, U2, LU2, U, mrs = scratch_moms

        a1, a2, a3, a4 = self.bdf3_A
        g1, g2, g3  = self.bdf3_G
        b = self.bdf3_B

        # if this is the first step, use the ars-222 scheme
        if self.nacptsteps == 0:
            tloc, dtloc = t, dt/self.nstepsInit

            fno1 = []
            for T, _ in enumerate(zip(sys.ele_types)):
                fno1.append(sys.eles_scal_upts_inb_full[T][fn].get())

            for step in range(self.nstepsInit):
                fn2 = super().step(tloc, dtloc);
                tloc += dtloc

            fno2 = []
            for T, _ in enumerate(zip(sys.ele_types)):
                fno2.append(sys.eles_scal_upts_inb_full[T][fn2].get())

            for step in range(self.nstepsInit):
                super().step(tloc, dtloc);
                tloc += dtloc

            for T, _ in enumerate(zip(sys.ele_types)):
                sys.eles_scal_upts_inb_full[T][f1].set(fno1[T])

            moment(t, f1, U1, frs); self.explicit(t, f1, L1, r0, r1); 
            moment(t, L1, LU1, frs);

            for T, _ in enumerate(zip(sys.ele_types)):
                sys.eles_scal_upts_inb_full[T][f2].set(fno2[T])
            moment(t, f2, U2, frs); self.explicit(t, f2, L2, r0, r1); 
            moment(t, L2, LU2, frs);

            del fno, fno2; return fn

        # if this is the second step, use the ars-222 scheme
        if self.nacptsteps == 1: return

        # Compute the moment of the initial distribution
        copym(U0, U1); copym(U1, U2);
        moment(t, fn, U2, frs);

        # Compute the explicit part; L2 = -∇·f(fn);
        copyd(L0, L1); copyd(L1, L2);
        self.explicit(t, fn, L2, r0, r1)

        # Compute the moment of the explicit part
        copym(LU0, LU1); copym(LU1, LU2); 
        moment(t, L2, LU2, frs)

        # update the moments
        updateMoment(dt, a1, U0, -g1, LU0, a2, U1, -g2, LU1, 
            a3, U2, -g3, LU2, a4, U, b)

        # implictly construct the Maxwellian (or Gaussian, etc.) given moments
        consMaxwellian(t, U, M, mrs)

        # update the distribution
        copyd(f0, f1); copyd(f1, f2); copyd(f2, fn);
        updateDist(dt, a1, f0, -g1, L0, a2, f1, -g2, L1, a3, f2, -g3, L2, 
            b, M, a4, U, fn)

        return fn