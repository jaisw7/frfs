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
        copyto, cofrfsom = self._copy_to_reg_dgfs, self._copy_from_reg_dgfs
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
        #    cofrfsom(r0, r1, 0, NvBS-1, sidx, eidx-1)

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
            cofrfsom(r0, r0, 0, NvBS-1, sidx, eidx-1)

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
        copyto, cofrfsom = self._copy_to_reg_dgfs, self._copy_from_reg_dgfs
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
        #    cofrfsom(r0, r1, 0, NvBS-1, sidx, eidx-1)

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
            cofrfsom(r1, r1, 0, NvBS-1, sidx, eidx-1)

        # Second stage; r2 = -∇·f(r1); r0 = 0.5*r0 + 0.5*r1 + 0.5*dt*r2
        # call collide
        collide(t+dt, r1, r2) 

        # multiply by inverse mass
        #for batch in range(vm.NvBatches()):
        #    sidx, eidx = batch*NvBS, (batch+1)*NvBS
        #    copyto(r2, r2, sidx, eidx-1, 0, NvBS-1)            
        #    invmass(r2, r1)
        #    cofrfsom(r1, r2, 0, NvBS-1, sidx, eidx-1)

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
            cofrfsom(r0, r0, 0, NvBS-1, sidx, eidx-1)


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
        copyto, cofrfsom = self._copy_to_reg_dgfs, self._copy_from_reg_dgfs
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
        #        cofrfsom(r0, r0+1, 0, NvBS-1, sidx, eidx-1)

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
                cofrfsom(r0, r0, 0, NvBS-1, sidx, eidx-1)

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
        copyto, cofrfsom = self._copy_to_reg_dgfs, self._copy_from_reg_dgfs
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
        #        cofrfsom(r0, r0+1, 0, NvBS-1, sidx, eidx-1)

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
                cofrfsom(r0+1, r0+1, 0, NvBS-1, sidx, eidx-1)


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
        #        cofrfsom(r0+1, r0+2, 0, NvBS-1, sidx, eidx-1)

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
                cofrfsom(r0, r0, 0, NvBS-1, sidx, eidx-1)

        return 0
