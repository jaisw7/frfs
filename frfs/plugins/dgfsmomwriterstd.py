# -*- coding: utf-8 -*-

from frfs.inifile import Inifile
from frfs.plugins.base import BasePlugin
from frfs.writers.native import NativeWriter
import numpy as np
import warnings

class DGFSMomWriterStdPlugin(BasePlugin):
    name = 'dgfsmomwriterstd'
    systems = ['dgfs', 'adgfs']
    formulations = ['std']

    def _compute_moments_1D(self, intg):
        vm = intg.system.vm
        cv = vm.cv()
        vsize = vm.vsize()
        cw = vm.cw()
        T0 = vm.T0()
        rho0 = vm.rho0()
        molarMass0 = vm.molarMass0()
        u0 = vm.u0()
        mr = molarMass0/molarMass0
        mcw = mr*cw

        # Note: I assume that the soln is stored in the first register
        # This won't affect bulk properties very substantially 
        full_soln = [eb[intg._idxcurr].get() 
            for eb in intg.system.eles_scal_upts_inb_full]

        for i in range(len(self._bulksoln)): 
            ele_sol, soln = self._bulksoln[i], full_soln[i]
            nupts, nvar, neles = ele_sol.shape
            ele_sol.fill(0)

            #[upts, var, ele]
            #non-dimensional mass density
            ele_sol[:,0,:] = np.sum(soln, axis=1)*mcw

            if(np.sum(ele_sol[:,0,:])) < 1e-10:
                warnings.warn("density below 1e-10", RuntimeWarning)
                continue

            #non-dimensional velocities
            ele_sol[:,1,:] = np.tensordot(soln, cv[0,:], axes=(1,0))*mcw
            ele_sol[:,1,:] /= ele_sol[:,0,:]
            ele_sol[:,2,:] = np.tensordot(soln, cv[1,:], axes=(1,0))*mcw
            ele_sol[:,2,:] /= ele_sol[:,0,:]

            # peculiar velocity
            cx = cv[0,:].reshape((1,vsize,1))-ele_sol[:,1,:].reshape(
                (nupts,1,neles))
            cy = cv[1,:].reshape((1,vsize,1))-ele_sol[:,2,:].reshape(
                (nupts,1,neles))
            cz = cv[2,:].reshape((1,vsize,1))-np.zeros((nupts,1,neles))
            cSqr = cx*cx + cy*cy + cz*cz

            # non-dimensional temperature
            ele_sol[:,3,:] = np.sum(soln*cSqr, axis=1)*(2.0/3.0*mcw*mr)
            ele_sol[:,3,:] /= ele_sol[:,0,:]

            # non-dimensional heat-flux
            #ele_sol[:,4,:] = mr*np.tensordot(
            #    soln*cSqr, cv[0,:], axes=(1,0))*mcw
            ele_sol[:,4,:] = mr*np.sum(soln*cSqr*cx, axis=1)*mcw

            # dimensional rho, ux, uy, T, qx
            ele_sol[:,0:5,:] *= np.array([
                rho0, u0, u0, T0, 0.5*rho0*(u0**3)]).reshape(1,5,1)

            # dimensional pressure
            ele_sol[:,5,:] = (
                (mr*vm.R0/molarMass0)*ele_sol[:,0,:]*ele_sol[:,3,:])


    def _compute_moments_2D(self, intg):
        vm = intg.system.vm
        cv = vm.cv()
        vsize = vm.vsize()
        cw = vm.cw()
        T0 = vm.T0()
        rho0 = vm.rho0()
        molarMass0 = vm.molarMass0()
        u0 = vm.u0()
        mr = molarMass0/molarMass0
        mcw = mr*cw

        full_soln = [eb[intg._idxcurr].get() 
            for eb in intg.system.eles_scal_upts_inb_full]

        for i in range(len(self._bulksoln)): 
            ele_sol, soln = self._bulksoln[i], full_soln[i]
            nupts, nvar, neles = ele_sol.shape
            ele_sol.fill(0)

            #non-dimensional mass density
            ele_sol[:,0,:] = np.sum(soln, axis=1)*mcw

            if(np.sum(ele_sol[:,0,:])) < 1e-10:
                warnings.warn("density below 1e-10", RuntimeWarning)
                continue

            #non-dimensional velocities
            ele_sol[:,1,:] = np.tensordot(soln, cv[0,:], axes=(1,0))*mcw
            ele_sol[:,1,:] /= ele_sol[:,0,:]
            ele_sol[:,2,:] = np.tensordot(soln, cv[1,:], axes=(1,0))*mcw
            ele_sol[:,2,:] /= ele_sol[:,0,:]

            # peculiar velocity
            cx = cv[0,:].reshape((1,vsize,1))-ele_sol[:,1,:].reshape(
                (nupts,1,neles))
            cy = cv[1,:].reshape((1,vsize,1))-ele_sol[:,2,:].reshape(
                (nupts,1,neles))
            cz = cv[2,:].reshape((1,vsize,1))-np.zeros((nupts,1,neles))
            cSqr = cx*cx + cy*cy + cz*cz

            # non-dimensional temperature
            ele_sol[:,3,:] = np.sum(soln*cSqr, axis=1)*(2.0/3.0*mcw*mr)
            ele_sol[:,3,:] /= ele_sol[:,0,:]

            # non-dimensional heat-flux
            ele_sol[:,4,:] = mr*np.sum(soln*cSqr*cx, axis=1)*mcw
            ele_sol[:,5,:] = mr*np.sum(soln*cSqr*cy, axis=1)*mcw

            # non-dimensional pressure-tensor components
            ele_sol[:,6,:] = 2*mr*np.sum(soln*cx*cx, axis=1)*mcw
            ele_sol[:,7,:] = 2*mr*np.sum(soln*cx*cy, axis=1)*mcw
            ele_sol[:,8,:] = 2*mr*np.sum(soln*cy*cy, axis=1)*mcw

            # dimensional rho, ux, uy, T, qx, qy, Pxx, Pxy, Pxx
            ele_sol[:,0:9,:] *= np.array([
                rho0, u0, u0, T0, 
                0.5*rho0*(u0**3), 0.5*rho0*(u0**3),
                0.5*rho0*(u0**2), 0.5*rho0*(u0**2), 0.5*rho0*(u0**2) 
            ]).reshape(1,9,1)

            # dimensional pressure
            ele_sol[:,9,:] = (
                (mr*vm.R0/molarMass0)*ele_sol[:,0,:]*ele_sol[:,3,:])


    def _compute_moments_3D(self, intg):
        vm = intg.system.vm
        cv = vm.cv()
        vsize = vm.vsize()
        cw = vm.cw()
        T0 = vm.T0()
        rho0 = vm.rho0()
        molarMass0 = vm.molarMass0()
        u0 = vm.u0()
        mr = molarMass0/molarMass0
        mcw = mr*cw

        full_soln = [eb[intg._idxcurr].get() 
            for eb in intg.system.eles_scal_upts_inb_full]

        for i in range(len(self._bulksoln)): 
            ele_sol, soln = self._bulksoln[i], full_soln[i]
            nupts, nvar, neles = ele_sol.shape
            ele_sol.fill(0)

            #[upts, var, ele]
            #non-dimensional mass density
            ele_sol[:,0,:] = np.sum(soln, axis=1)*mcw

            if(np.sum(ele_sol[:,0,:])) < 1e-10:
                warnings.warn("density below 1e-10", RuntimeWarning)
                continue

            #non-dimensional velocities
            ele_sol[:,1,:] = np.tensordot(soln, cv[0,:], axes=(1,0))*mcw
            ele_sol[:,1,:] /= ele_sol[:,0,:]
            ele_sol[:,2,:] = np.tensordot(soln, cv[1,:], axes=(1,0))*mcw
            ele_sol[:,2,:] /= ele_sol[:,0,:]
            ele_sol[:,3,:] = np.tensordot(soln, cv[2,:], axes=(1,0))*mcw
            ele_sol[:,3,:] /= ele_sol[:,0,:]

            # peculiar velocity
            cx = cv[0,:].reshape((1,vsize,1))-ele_sol[:,1,:].reshape(
                (nupts,1,neles))
            cy = cv[1,:].reshape((1,vsize,1))-ele_sol[:,2,:].reshape(
                (nupts,1,neles))
            cz = cv[2,:].reshape((1,vsize,1))-ele_sol[:,3,:].reshape(
                (nupts,1,neles))
            cSqr = cx*cx + cy*cy + cz*cz

            # non-dimensional temperature
            ele_sol[:,4,:] = np.sum(soln*cSqr, axis=1)*(2.0/3.0*mcw*mr)
            ele_sol[:,4,:] /= ele_sol[:,0,:]

            # non-dimensional heat-flux
            ele_sol[:,5,:] = mr*np.sum(soln*cSqr*cx, axis=1)*mcw
            ele_sol[:,6,:] = mr*np.sum(soln*cSqr*cy, axis=1)*mcw
            ele_sol[:,7,:] = mr*np.sum(soln*cSqr*cz, axis=1)*mcw

            # non-dimensional pressure-tensor components
            ele_sol[:,8,:] = 2*mr*np.sum(soln*cx*cx, axis=1)*mcw
            ele_sol[:,9,:] = 2*mr*np.sum(soln*cx*cy, axis=1)*mcw
            ele_sol[:,10,:] = 2*mr*np.sum(soln*cx*cz, axis=1)*mcw
            ele_sol[:,11,:] = 2*mr*np.sum(soln*cy*cy, axis=1)*mcw
            ele_sol[:,12,:] = 2*mr*np.sum(soln*cy*cz, axis=1)*mcw
            ele_sol[:,13,:] = 2*mr*np.sum(soln*cz*cz, axis=1)*mcw

            # dimensional rho, ux, uy, uz, T, qx, qy, qz, 6 pressure tensors
            ele_sol[:,0:14,:] *= np.array([
                rho0, u0, u0, u0, T0, 
                0.5*rho0*(u0**3), 0.5*rho0*(u0**3), 0.5*rho0*(u0**3),
                0.5*rho0*(u0**2), 0.5*rho0*(u0**2), 0.5*rho0*(u0**2),
                0.5*rho0*(u0**2), 0.5*rho0*(u0**2), 0.5*rho0*(u0**2)
            ]).reshape(1,14,1)

            # dimensional pressure
            ele_sol[:,14,:] = (
                (mr*vm.R0/molarMass0)*ele_sol[:,0,:]*ele_sol[:,4,:])


    _moment_maps = {
        1: _compute_moments_1D, 
        2: _compute_moments_2D, 
        3: _compute_moments_3D
    }

    @property
    def bulksoln(self): return self._bulksoln

    def __init__(self, intg, cfgsect, suffix=None, write=True):
        super().__init__(intg, cfgsect, suffix)

        # Construct the solution writer
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')

        # fix nvars: the bulk properties
        # these variables are same as in DGFSElements
        privarmap = {
            1: ['rho', 'U:x', 'U:y', 'T', 'Q:x', 'p'],
            2: ['rho', 'U:x', 'U:y', 'T', 'Q:x', 'Q:y', 
                'P:xx', 'P:xy', 'P:yy', 'p'],
            3: ['rho', 'U:x', 'U:y', 'U:z', 'T', 'Q:x', 'Q:y', 'Q:z', 
                'P:xx', 'P:xy', 'P:xz', 'P:yy', 'P:yz', 'P:zz', 'p']
        }
        convarmap = privarmap
        
        self.nvars = len(privarmap[self.ndims])

        self._writer = NativeWriter(intg, self.nvars, basedir, basename,
                                    prefix='moments')

        # Output time step and next output time
        if write: 
            self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_next = intg.tcurr

        # Output field names
        #self.fields = intg.system.elementscls.convarmap[self.ndims]
        self.fields = convarmap[self.ndims]

        # function maps 
        self._compute_moments = self._moment_maps[self.ndims]

        # debug
        #for item in intg.soln:
        #    print(item.shape)
        #ele_map = intg.system.ele_map
        #for k, ele in elemap.items():
        #    np.empty((ele.nupts, ele.nvars, ele.neles))

        # allocate variables
        self._bulksoln = [np.empty((item.shape[0], self.nvars, item.shape[2])) 
            for item in intg.soln]

        # Register our output times with the integrator
        if write:
            intg.call_plugin_dt(self.dt_out)
            self(intg) # helps us to compute moments from restart file

        # If we're not restarting then write out the initial solution
        #if not intg.isrestart:
        #    self(intg)
        #else:
        #    self.tout_next += self.dt_out       

    def compute_moments(self, intg):
        self._compute_moments(self, intg)

    def __call__(self, intg):
        if abs(self.tout_next - intg.tcurr) > self.tol:
            return

        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        stats.set('data', 'prefix', 'moments')
        intg.collect_stats(stats)

        # Prepare the metadata
        metadata = dict(intg.cfgmeta,
                        stats=stats.tostr(),
                        mesh_uuid=intg.mesh_uuid)

        # compute the moments
        self._compute_moments(self, intg)
        #self.(self._moment_maps[self.ndims])(intg)

        # Write out the file
        solnfname = self._writer.write(self._bulksoln, metadata, intg.tcurr)

        # If a post-action has been registered then invoke it
        self._invoke_postaction(mesh=intg.system.mesh.fname, soln=solnfname,
                                t=intg.tcurr)

        # Compute the next output time
        self.tout_next = intg.tcurr + self.dt_out

