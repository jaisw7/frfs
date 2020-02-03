# -*- coding: utf-8 -*-

from frfs.solvers.base import BaseSystem
from frfs.solvers.dgfsbi.elements import DGFSBiElements
from frfs.solvers.dgfsbi.velocitymesh import DGFSBiVelocityMesh
from frfs.solvers.dgfsbi.scattering import DGFSBiScatteringModel
from frfs.solvers.dgfsbi.initcond import DGFSBiInitCondition
from frfs.solvers.dgfsbi.inters import (DGFSBiIntInters, DGFSBiMPIInters,
                                       DGFSBiBCInters)
from frfs.inifile import Inifile
from frfs.util import subclass_where, proxylist, ndrange
import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray
from frfs.mpiutil import get_comm_rank_root, get_mpi

class DGFSBiSystem(BaseSystem):
    #print("Initializing DGFSBi system")

    name = 'dgfsbi'

    elementscls = DGFSBiElements
    intinterscls = DGFSBiIntInters
    mpiinterscls = DGFSBiMPIInters
    bbcinterscls = DGFSBiBCInters
    velocitymeshcls = DGFSBiVelocityMesh

    _nqueues = 2

    # need to fix (intg._stepper_nregs, intg._stepper_nregs_orig)
    _nspcs = 2

    def __init__(self, backend, rallocs, mesh, initsoln, nreg, cfg):

        if(not backend.name=='cuda'):
            raise ValueError("CUDA backend supported!")

        # load the velocity mesh
        self.vm = self.velocitymeshcls(backend, cfg, self._nspcs)

        cv = self.vm.cv()
        vsize = self.vm.vsize()

        # need to define the expressions
        # the prefix "f_" should be same as in elementcls distvar
        # size of distvar should be equal to NvBatchSize
        for ivar in range(self.vm.NvBatchSize()):
            cfg.set('soln-ics', 'f_' + str(ivar), '0.')
        
        # now, we can initialize things
        super().__init__(backend, rallocs, mesh, initsoln, nreg, cfg, 
            vm=self.vm)
        print('Finished initializing the BaseSystem')


        # define the time-step
        minjac = 100.0
        for t, ele in self.ele_map.items():
            djac = ele.djac_at_np('upts')
            minjac = np.min([minjac, np.min(djac)])
        advmax = self.vm.L()
        unitCFLdt = np.array([np.sqrt(minjac)/advmax/self.ndims])
        gunitCFLdt = np.zeros(1)
        # MPI info
        comm, rank, root = get_comm_rank_root()
        # Reduce and, if we are the root rank, output
        if rank != root:
            comm.Reduce(unitCFLdt, gunitCFLdt, op=get_mpi('min'), root=root)
        else:
            comm.Reduce(unitCFLdt, gunitCFLdt, op=get_mpi('min'), root=root)
            print("Time-step for unit CFL:", gunitCFLdt)
            print("The actual time-step will depend on DG order CFL")


        # load the scattering model
        smn = cfg.get('scattering-model', 'type')
        scatteringcls = subclass_where(DGFSBiScatteringModel, 
            scattering_model=smn)
        self.sm = scatteringcls(backend, self.cfg, self.vm)


        # Allocate and bank the storage required by the time integrator
        #eles_scal_upts_full = proxylist(self.ele_banks)
        eles_scal_upts_inb_full = proxylist(self.ele_banks)
        spcs_eles_scal_upts_full = [list(self.ele_banks) 
                    for spcs in range(self._nspcs)]
        
        if initsoln:
            #raise ValueError("Not implemented")

            # Load the config and stats files from the solution
            solncfg = Inifile(initsoln['config'])
            solnsts = Inifile(initsoln['stats'])

            # Get the names of the conserved variables (fields)
            solnfields = solnsts.get('data', 'fields', '')
            # see dgfsdistwriterbi.py plugin
            currfields = []
            fields = ['f_'+str(i) for i in range(vsize)]
            lf = len(fields)
            for p in range(self._nspcs):
                currfields.extend(fields)
                for ivar in range(-1,-lf-1,-1): 
                    currfields[ivar] += ':'+str(p+1)
            currfields = ','.join(currfields)

            # Ensure they match up
            if solnfields and solnfields != currfields:
                raise RuntimeError('Invalid solution for system')

            # Ensure the solnfields are not empty
            if not solnfields:
                raise RuntimeError('Invalid solution for system')

            nreg0 = nreg//self._nspcs
            assert nreg==nreg0*self._nspcs, "Should be multiple of nspcs"

            # Process the solution
            for t, (k, ele) in enumerate(self.ele_map.items()):
                soln = initsoln['soln_%s_p%d' % (k, rallocs.prank)]
                
                #ele.set_ics_from_soln(soln, solncfg)
                # Recreate the existing solution basis
                solnb = ele.basis.__class__(None, solncfg)

                # Form the interpolation operator
                interp = solnb.ubasis.nodal_basis_at(ele.basis.upts)

                # Apply and reshape
                data = np.dot(interp, soln.reshape(solnb.nupts, -1))
                data = data.reshape(ele.nupts, self._nspcs*vsize, ele.neles)
                
                for p in range(self._nspcs):
                    spcs_eles_scal_upts_full[p][t] = data[:, 
                        p*vsize:(p+1)*vsize, :]
        else:
            # load the initial condition model
            icn = cfg.get('soln-ics', 'type')
            initcondcls = subclass_where(DGFSBiInitCondition, model=icn)
            ic = initcondcls(backend, cfg, self.vm, 'soln-ics')
            #initvals = ic.get_init_vals()

            nreg0 = nreg//self._nspcs
            assert nreg==nreg0*self._nspcs, "Should be multiple of nspcs"

            # loop over the sub-domains in the full mixed domain
            for p in range(self._nspcs):
                for t, ele in enumerate(self.ele_map.values()):
                    spcs_eles_scal_upts_full[p][t] = np.empty(
                        (ele.nupts, vsize, ele.neles))
                    
                    ic.apply_init_vals(p, spcs_eles_scal_upts_full[p][t], ele)
                    # Convert from primitive to conservative form if needed
                    

        nreg0 = nreg//self._nspcs
        assert nreg==nreg0*self._nspcs, "Should be multiple of nspcs"

        for t in range(len(eles_scal_upts_inb_full)):
            scal_upts_full = []
            for p in range(self._nspcs):    
                if p==0:
                    scal_upts_full = [
                        backend.matrix(spcs_eles_scal_upts_full[p][t].shape,
                        spcs_eles_scal_upts_full[p][t], tags={'align'}) 
                        for i in range(nreg0)]
                else:
                    scal_upts_full.extend([
                        backend.matrix(spcs_eles_scal_upts_full[p][t].shape,
                        spcs_eles_scal_upts_full[p][t], tags={'align'}) 
                        for i in range(nreg0)])
            
            eles_scal_upts_inb_full[t] = backend.matrix_bank(scal_upts_full)
            #eles_scal_upts_outb_full[t] = backend.matrix_bank(scal_upts_full)

        self.eles_scal_upts_inb_full = eles_scal_upts_inb_full
        del spcs_eles_scal_upts_full


    def rhs(self, t, idx, uinbank, foutbank, batch, varidx):
        runall = self.backend.runall
        q1, q2 = self._queues
        kernels = self._kernels

        self.eles_scal_upts_inb.active = uinbank
        self.eles_scal_upts_outb.active = foutbank

        dtype = self.eles_scal_upts_inb[0][0].dtype
        varidx_f = np.array([varidx], dtype=dtype)[0]
        
        q1 << kernels['eles', 'disu']()
        q1 << kernels['mpiint', 'scal_fpts_pack']()
        runall([q1])

        if ('eles', 'copy_soln') in kernels:
            q1 << kernels['eles', 'copy_soln']()
        q1 << kernels['eles', 'tdisf'](varidx=varidx_f)
        q1 << kernels['eles', 'tdivtpcorf']()
        q1 << kernels['iint', 'comm_flux'](varidx=varidx_f)
        q1 << kernels['bcint', 'comm_flux'+idx](t=t, varidx=varidx_f)

        q2 << kernels['mpiint', 'scal_fpts_send']()
        q2 << kernels['mpiint', 'scal_fpts_recv']()
        q2 << kernels['mpiint', 'scal_fpts_unpack']()

        runall([q1, q2])

        q1 << kernels['mpiint', 'comm_flux'](varidx=varidx_f)
        q1 << kernels['eles', 'tdivtconf']()
        if ('eles', 'tdivf_qpts') in kernels:
            q1 << kernels['eles', 'tdivf_qpts']()
            q1 << kernels['eles', 'negdivconf'](t=t)
            q1 << kernels['eles', 'divf_upts']()
        else:
            q1 << kernels['eles', 'negdivconf'](t=t)

        runall([q1])


    def updatebc(self, t, idx, uinbank, foutbank, batch, varidx):
        runall = self.backend.runall
        q1, q2 = self._queues
        kernels = self._kernels
        
        dtype = self.eles_scal_upts_inb[0][0].dtype
        varidx_f = np.array([varidx], dtype=dtype)[0]

        self.eles_scal_upts_inb.active = uinbank
        self.eles_scal_upts_outb.active = foutbank

        # compute the face-point soln given the initial soln 
        q1 << kernels['eles', 'disu']()
        runall([q1])

        q1 << kernels['bcint', 'update_bc'+idx](t=t, varidx=varidx_f)
        runall([q1])


    def collide(self, t, regids):
        fsoln = self.eles_scal_upts_inb_full
        #print(regids)
        for e, ele in enumerate(self.ele_types):
            arr = list(regids)
            for i, regid in enumerate(regids):
                fsoln.active = regid
                arr[i] = fsoln[e][regid]
            nupts, _, _ = arr[0].traits
            _, neles = arr[0].ioshape[1:]

            for elem, upt in ndrange(neles, nupts):
                self.sm.fs(elem, upt, *arr)

    def invmass(self, uinbank, foutbank):
        runall = self.backend.runall
        q1, q2 = self._queues
        kernels = self._kernels
        
        self.eles_scal_upts_inb.active = uinbank
        self.eles_scal_upts_outb.active = foutbank

        # multiply by the inverse mass matrix
        q1 << kernels['eles', 'invmass']()
        runall([q1])

