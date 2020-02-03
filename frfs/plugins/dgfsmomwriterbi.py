# -*- coding: utf-8 -*-

from frfs.inifile import Inifile
from frfs.plugins.base import BasePlugin
from frfs.writers.native import NativeWriter
import numpy as np
import warnings

class DGFSMomWriterBiPlugin(BasePlugin):
    name = 'dgfsmomwriterbi'
    systems = ['dgfsbi']
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
        mr_ = vm.masses()
        nspcs = vm.nspcs()
        mcw_ = [mr_[p]*cw for p in range(nspcs)]
        nregs = intg._stepper_nregs_orig
        nprops = self.nvars//nspcs

        for propt in self._bulksolntot:
            propt.fill(0.)
        propt = self._bulksolntot

        full_soln = [0]*nspcs
        for p in range(nspcs):
            # Note: I assume that the soln is stored in the first register
            full_soln[p] = [eb[nregs*p].get() 
                for eb in intg.system.eles_scal_upts_inb_full]

        for p in range(nspcs):
            #r0 = nregs*p
            mr = mr_[p]
            mcw = mr*cw
            ispcs = nprops*p
            for i in range(len(self._bulksoln)): 
                ele_sol, soln = self._bulksoln[i], full_soln[i]
                nupts, nvar, neles = ele_sol.shape
                if p==0: ele_sol.fill(0)

                #[upts, var, ele]
                #non-dimensional mass density
                ele_sol[:,ispcs+0,:] = np.sum(soln, axis=1)*mcw

                if(np.sum(ele_sol[:,ispcs+0,:])) < 1e-10:
                    warnings.warn("density below 1e-10", RuntimeWarning)
                    continue

                #non-dimensional velocities
                ele_sol[:,ispcs+1,:] = np.tensordot(soln, cv[0,:], 
                    axes=(1,0))*mcw
                ele_sol[:,ispcs+1,:] /= ele_sol[:,ispcs+0,:]
                ele_sol[:,ispcs+2,:] = np.tensordot(soln, cv[1,:], 
                    axes=(1,0))*mcw
                ele_sol[:,ispcs+2,:] /= ele_sol[:,ispcs+0,:]

                # peculiar velocity for species
                cx = cv[0,:].reshape((1,vsize,1))-ele_sol[:,ispcs+1,:].reshape(
                    (nupts,1,neles))
                cy = cv[1,:].reshape((1,vsize,1))-ele_sol[:,ispcs+2,:].reshape(
                    (nupts,1,neles))
                cz = cv[2,:].reshape((1,vsize,1))-np.zeros((nupts,1,neles))
                cSqr = cx*cx + cy*cy + cz*cz

                # non-dimensional temperature
                ele_sol[:,ispcs+3,:] = np.sum(soln*cSqr, 
                    axis=1)*(2.0/3.0*mcw*mr)
                ele_sol[:,ispcs+3,:] /= ele_sol[:,ispcs+0,:]

                # total mass density
                propt[i][:,0,:] += ele_sol[:,ispcs+0,:]

                # total velocity
                propt[i][:,1,:] += ele_sol[:,ispcs+0,:]*ele_sol[:,ispcs+1,:]
                propt[i][:,2,:] += ele_sol[:,ispcs+0,:]*ele_sol[:,ispcs+2,:]


        for p in range(nspcs):
            #r0 = nregs*p
            mr = mr_[p]
            mcw = mr*cw
            ispcs = nprops*p
            for i in range(len(self._bulksoln)): 
                if(p==0):
                    if(np.sum(propt[i][:,0,:])) < 1e-10:
                        warnings.warn("density below 1e-10", RuntimeWarning)
                        continue

                    # normalize the total velocity
                    propt[i][:,1,:] /= propt[i][:,0,:]
                    propt[i][:,2,:] /= propt[i][:,0,:]
                    
                ele_sol, soln = self._bulksoln[i], full_soln[i]
                nupts, nvar, neles = ele_sol.shape

                # peculiar velocity
                cx = cv[0,:].reshape((1,vsize,1))-propt[i][:,1,:].reshape(
                    (nupts,1,neles))
                cy = cv[1,:].reshape((1,vsize,1))-propt[i][:,2,:].reshape(
                    (nupts,1,neles))
                cz = cv[2,:].reshape((1,vsize,1))-np.zeros((nupts,1,neles))
                cSqr = cx*cx + cy*cy + cz*cz

                # non-dimensional heat-flux
                #ele_sol[:,4,:] = mr*np.tensordot(
                #    soln*cSqr, cv[0,:], axes=(1,0))*mcw
                ele_sol[:,ispcs+4,:] = mr*np.sum(soln*cSqr*cx, axis=1)*mcw

                # dimensional rho, ux, uy, T, qx
                ele_sol[:,ispcs+0:ispcs+5,:] *= np.array([
                    rho0, u0, u0, T0, 0.5*rho0*(u0**3)]).reshape(1,5,1)

                # dimensional pressure
                ele_sol[:,ispcs+5,:] = (
                    (mr*vm.R0/molarMass0)
                    *ele_sol[:,ispcs+0,:]*ele_sol[:,ispcs+3,:])

                # dimensional number density
                ele_sol[:,ispcs+6,:] = (
                    (vm.NA/mr/molarMass0)*ele_sol[:,ispcs+0,:])

        del full_soln


    def _compute_moments_2D(self, intg):
        vm = intg.system.vm
        cv = vm.cv()
        vsize = vm.vsize()
        cw = vm.cw()
        T0 = vm.T0()
        rho0 = vm.rho0()
        molarMass0 = vm.molarMass0()
        u0 = vm.u0()
        mr_ = vm.masses()
        nspcs = vm.nspcs()
        mcw_ = [mr_[p]*cw for p in range(nspcs)]
        nregs = intg._stepper_nregs_orig
        nprops = self.nvars//nspcs

        for propt in self._bulksolntot:
            propt.fill(0.)
        propt = self._bulksolntot

        full_soln = [0]*nspcs
        for p in range(nspcs):
            # Note: I assume that the soln is stored in the first register
            full_soln[p] = [eb[nregs*p].get() 
                for eb in intg.system.eles_scal_upts_inb_full]

        # compute massrho, velocity, temperature
        for p in range(nspcs):
            mr = mr_[p]
            mcw = mr*cw
            ispcs = nprops*p
            for i in range(len(self._bulksoln)): 
                ele_sol, soln = self._bulksoln[i], full_soln[p][i]
                nupts, nvar, neles = ele_sol.shape
                if p==0: ele_sol.fill(0)

                #[upts, var, ele]
                #non-dimensional mass density
                ele_sol[:,ispcs+0,:] = np.sum(soln, axis=1)*mcw

                if(np.sum(ele_sol[:,ispcs+0,:])) < 1e-10:
                    warnings.warn("density below 1e-10", RuntimeWarning)
                    continue

                #non-dimensional velocities
                ele_sol[:,ispcs+1,:] = np.tensordot(soln, cv[0,:], 
                    axes=(1,0))*mcw
                ele_sol[:,ispcs+1,:] /= ele_sol[:,ispcs+0,:]
                ele_sol[:,ispcs+2,:] = np.tensordot(soln, cv[1,:], 
                    axes=(1,0))*mcw
                ele_sol[:,ispcs+2,:] /= ele_sol[:,ispcs+0,:]

                # peculiar velocity for species
                cx = cv[0,:].reshape((1,vsize,1))-ele_sol[:,ispcs+1,:].reshape(
                    (nupts,1,neles))
                cy = cv[1,:].reshape((1,vsize,1))-ele_sol[:,ispcs+2,:].reshape(
                    (nupts,1,neles))
                cz = cv[2,:].reshape((1,vsize,1))-np.zeros((nupts,1,neles))
                cSqr = cx*cx + cy*cy + cz*cz

                # non-dimensional temperature
                ele_sol[:,ispcs+3,:] = np.sum(soln*cSqr, 
                    axis=1)*(2.0/3.0*mcw*mr)
                ele_sol[:,ispcs+3,:] /= ele_sol[:,ispcs+0,:]

                # total mass density
                propt[i][:,0,:] += ele_sol[:,ispcs+0,:]

                # total velocity
                propt[i][:,1,:] += ele_sol[:,ispcs+0,:]*ele_sol[:,ispcs+1,:]
                propt[i][:,2,:] += ele_sol[:,ispcs+0,:]*ele_sol[:,ispcs+2,:]


        for p in range(nspcs):
            mr = mr_[p]
            mcw = mr*cw
            ispcs = nprops*p
            for i in range(len(self._bulksoln)): 
                if(p==0):
                    if(np.sum(propt[i][:,0,:])) < 1e-10:
                        warnings.warn("density below 1e-10", RuntimeWarning)
                        continue

                    # normalize the total velocity
                    propt[i][:,1,:] /= propt[i][:,0,:]
                    propt[i][:,2,:] /= propt[i][:,0,:]
                    
                ele_sol, soln = self._bulksoln[i], full_soln[p][i]
                nupts, nvar, neles = ele_sol.shape

                # peculiar velocity
                cx = cv[0,:].reshape((1,vsize,1))-propt[i][:,1,:].reshape(
                    (nupts,1,neles))
                cy = cv[1,:].reshape((1,vsize,1))-propt[i][:,2,:].reshape(
                    (nupts,1,neles))
                cz = cv[2,:].reshape((1,vsize,1))-np.zeros((nupts,1,neles))
                cSqr = cx*cx + cy*cy + cz*cz

                # non-dimensional heat-flux
                ele_sol[:,ispcs+4,:] = mr*np.sum(soln*cSqr*cx, axis=1)*mcw
                ele_sol[:,ispcs+5,:] = mr*np.sum(soln*cSqr*cy, axis=1)*mcw

                # non-dimensional pressure-tensor components
                ele_sol[:,ispcs+6,:] = 2*mr*np.sum(soln*cx*cx, axis=1)*mcw
                ele_sol[:,ispcs+7,:] = 2*mr*np.sum(soln*cx*cy, axis=1)*mcw
                ele_sol[:,ispcs+8,:] = 2*mr*np.sum(soln*cy*cy, axis=1)*mcw

                # dimensional rho, ux, uy, T, qx, qy, Pxx, Pxy, Pxx
                ele_sol[:,ispcs+0:ispcs+9,:] *= np.array([
                    rho0, u0, u0, T0, 
                    0.5*rho0*(u0**3), 0.5*rho0*(u0**3),
                    0.5*rho0*(u0**2), 0.5*rho0*(u0**2), 0.5*rho0*(u0**2) 
                ]).reshape(1,9,1)

                # dimensional pressure
                ele_sol[:,ispcs+9,:] = (
                    (mr*vm.R0/molarMass0)*
                    ele_sol[:,ispcs+0,:]*ele_sol[:,ispcs+3,:])

                # dimensional number density
                ele_sol[:,ispcs+10,:] = (
                    (vm.NA/mr/molarMass0)*ele_sol[:,ispcs+0,:])

        del full_soln


    def _compute_moments_3D(self, intg):
        vm = intg.system.vm
        cv = vm.cv()
        vsize = vm.vsize()
        cw = vm.cw()
        T0 = vm.T0()
        rho0 = vm.rho0()
        molarMass0 = vm.molarMass0()
        u0 = vm.u0()
        mr_ = vm.masses()
        nspcs = vm.nspcs()
        mcw_ = [mr_[p]*cw for p in range(nspcs)]
        nregs = intg._stepper_nregs_orig
        nprops = self.nvars//nspcs

        for propt in self._bulksolntot:
            propt.fill(0.)
        propt = self._bulksolntot

        full_soln = [0]*nspcs
        for p in range(nspcs):
            # Note: I assume that the soln is stored in the first register
            full_soln[p] = [eb[nregs*p].get() 
                for eb in intg.system.eles_scal_upts_inb_full]

        for p in range(nspcs):
            mr = mr_[p]
            mcw = mr*cw
            ispcs = nprops*p
            for i in range(len(self._bulksoln)): 
                ele_sol, soln = self._bulksoln[i], full_soln[p][i]
                nupts, nvar, neles = ele_sol.shape
                if p==0: ele_sol.fill(0)

                #[upts, var, ele]
                #non-dimensional mass density
                ele_sol[:,ispcs+0,:] = np.sum(soln, axis=1)*mcw

                if(np.sum(ele_sol[:,ispcs+0,:])) < 1e-10:
                    warnings.warn("density below 1e-10", RuntimeWarning)
                    continue

                #non-dimensional velocities
                ele_sol[:,ispcs+1,:] = np.tensordot(soln, cv[0,:], 
                    axes=(1,0))*mcw
                ele_sol[:,ispcs+1,:] /= ele_sol[:,ispcs+0,:]
                ele_sol[:,ispcs+2,:] = np.tensordot(soln, cv[1,:], 
                    axes=(1,0))*mcw
                ele_sol[:,ispcs+2,:] /= ele_sol[:,ispcs+0,:]
                ele_sol[:,ispcs+3,:] = np.tensordot(soln, cv[2,:], 
                    axes=(1,0))*mcw
                ele_sol[:,ispcs+3,:] /= ele_sol[:,ispcs+0,:]

                # peculiar velocity for species
                cx = cv[0,:].reshape((1,vsize,1))-ele_sol[:,ispcs+1,:].reshape(
                    (nupts,1,neles))
                cy = cv[1,:].reshape((1,vsize,1))-ele_sol[:,ispcs+2,:].reshape(
                    (nupts,1,neles))
                cz = cv[2,:].reshape((1,vsize,1))-ele_sol[:,ispcs+3,:].reshape(
                    (nupts,1,neles))
                cSqr = cx*cx + cy*cy + cz*cz

                # non-dimensional temperature
                ele_sol[:,ispcs+4,:] = np.sum(soln*cSqr, 
                    axis=1)*(2.0/3.0*mcw*mr)
                ele_sol[:,ispcs+4,:] /= ele_sol[:,ispcs+0,:]

                # total mass density
                propt[i][:,0,:] += ele_sol[:,ispcs+0,:]

                # total velocity
                propt[i][:,1,:] += ele_sol[:,ispcs+0,:]*ele_sol[:,ispcs+1,:]
                propt[i][:,2,:] += ele_sol[:,ispcs+0,:]*ele_sol[:,ispcs+2,:]
                propt[i][:,3,:] += ele_sol[:,ispcs+0,:]*ele_sol[:,ispcs+3,:]


        for p in range(nspcs):
            mr = mr_[p]
            mcw = mr*cw
            ispcs = nprops*p
            for i in range(len(self._bulksoln)): 
                if(p==0):
                    if(np.sum(propt[i][:,0,:])) < 1e-10:
                        warnings.warn("density below 1e-10", RuntimeWarning)
                        continue                    
           
                    # normalize the total velocity
                    propt[i][:,1,:] /= propt[i][:,0,:]
                    propt[i][:,2,:] /= propt[i][:,0,:]
                    propt[i][:,3,:] /= propt[i][:,0,:]
                    
                ele_sol, soln = self._bulksoln[i], full_soln[p][i]
                nupts, nvar, neles = ele_sol.shape

                # peculiar velocity
                cx = cv[0,:].reshape((1,vsize,1))-propt[i][:,1,:].reshape(
                    (nupts,1,neles))
                cy = cv[1,:].reshape((1,vsize,1))-propt[i][:,2,:].reshape(
                    (nupts,1,neles))
                cz = cv[2,:].reshape((1,vsize,1))-propt[i][:,3,:].reshape(
                    (nupts,1,neles))
                cSqr = cx*cx + cy*cy + cz*cz

                # non-dimensional heat-flux
                ele_sol[:,ispcs+5,:] = mr*np.sum(soln*cSqr*cx, axis=1)*mcw
                ele_sol[:,ispcs+6,:] = mr*np.sum(soln*cSqr*cy, axis=1)*mcw
                ele_sol[:,ispcs+7,:] = mr*np.sum(soln*cSqr*cz, axis=1)*mcw

                # non-dimensional pressure-tensor components
                ele_sol[:,ispcs+8,:] = 2*mr*np.sum(soln*cx*cx, axis=1)*mcw
                ele_sol[:,ispcs+9,:] = 2*mr*np.sum(soln*cx*cy, axis=1)*mcw
                ele_sol[:,ispcs+10,:] = 2*mr*np.sum(soln*cx*cz, axis=1)*mcw
                ele_sol[:,ispcs+11,:] = 2*mr*np.sum(soln*cy*cy, axis=1)*mcw
                ele_sol[:,ispcs+12,:] = 2*mr*np.sum(soln*cy*cz, axis=1)*mcw
                ele_sol[:,ispcs+13,:] = 2*mr*np.sum(soln*cz*cz, axis=1)*mcw

                # dimensional rho, ux, uy, uz, T, qx, qy, qz, 6 pressure tensors
                ele_sol[:,ispcs+0:ispcs+14,:] *= np.array([
                    rho0, u0, u0, u0, T0, 
                    0.5*rho0*(u0**3), 0.5*rho0*(u0**3), 0.5*rho0*(u0**3),
                    0.5*rho0*(u0**2), 0.5*rho0*(u0**2), 0.5*rho0*(u0**2),
                    0.5*rho0*(u0**2), 0.5*rho0*(u0**2), 0.5*rho0*(u0**2)
                ]).reshape(1,14,1)

                # dimensional pressure
                ele_sol[:,ispcs+14,:] = (
                    (mr*vm.R0/molarMass0)*
                    ele_sol[:,ispcs+0,:]*ele_sol[:,ispcs+4,:])

                # dimensional number density
                ele_sol[:,ispcs+15,:] = (
                    (vm.NA/mr/molarMass0)*ele_sol[:,ispcs+0,:])

        del full_soln


    _moment_maps = {
        1: _compute_moments_1D, 
        2: _compute_moments_2D, 
        3: _compute_moments_3D
    }

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Construct the solution writer
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')

        # fix nvars: the bulk properties
        # these variables are same as in DGFSElements
        privarmap = {
            1: ['rho', 'Ux', 'Uy', 'T', 'Qx', 'p', 'nden'],
            2: ['rho', 'Ux', 'Uy', 'T', 'Qx', 'Qy', 
                'Pxx', 'Pxy', 'Pyy', 'p', 'nden'],
            3: ['rho', 'Ux', 'Uy', 'Uz', 'T', 'Qx', 'Qy', 'Qz', 
                'Pxx', 'Pxy', 'Pxz', 'Pyy', 'Pyz', 'Pzz', 'p', 'nden']
        }

        # add variables for different species
        for ndims in [1,2,3]:
            var = privarmap[ndims]
            lv = len(var)
            newvar = []
            for p in range(intg.system._nspcs):
                newvar.extend(var)
                for ivar in range(-1,-lv-1,-1): newvar[ivar] += ':'+str(p+1)
            privarmap[ndims] = newvar

        convarmap = privarmap        
        self.nvars = len(privarmap[self.ndims])

        self._writer = NativeWriter(intg, self.nvars, basedir, basename,
                                    prefix='moments')

        # Output time step and next output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_next = intg.tcurr

        # Output field names
        #self.fields = intg.system.elementscls.convarmap[self.ndims]
        self.fields = convarmap[self.ndims]

        # function maps 
        self._compute_moments = self._moment_maps[self.ndims]

        # allocate variables
        self._bulksoln = [np.empty((item.shape[0], self.nvars, item.shape[2])) 
            for item in intg.soln]

        # scratch storage for total bulk properties
        self._bulksolntot = [np.empty((item.shape[0], 4, item.shape[2])) 
            for item in intg.soln]        

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # If we're not restarting then write out the initial solution
        if not intg.isrestart:
            self(intg)
        else:
            self.tout_next += self.dt_out       

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

