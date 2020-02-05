# -*- coding: utf-8 -*-

from frfs.solvers.dgfs.system import DGFSSystem
from frfs.solvers.dgfs.elements import DGFSElements
from frfs.solvers.dgfs.velocitymesh import DGFSVelocityMesh
from frfs.solvers.adgfs.scattering import ADGFSScatteringModel
from frfs.solvers.dgfs.initcond import DGFSInitCondition
from frfs.solvers.dgfs.inters import (DGFSIntInters, DGFSMPIInters,
                                       DGFSBCInters)

from frfs.inifile import Inifile
from frfs.util import subclass_where, proxylist, ndrange
import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray
from frfs.mpiutil import get_comm_rank_root, get_mpi

class ADGFSSystem(DGFSSystem):
    name = 'adgfs'

    elementscls = DGFSElements
    intinterscls = DGFSIntInters
    mpiinterscls = DGFSMPIInters
    bbcinterscls = DGFSBCInters
    velocitymeshcls = DGFSVelocityMesh
    scatteringcls = ADGFSScatteringModel

    _nqueues = 2

    def __init__(self, backend, rallocs, mesh, initsoln, nreg, cfg):
        super().__init__(backend, rallocs, mesh, initsoln, nreg, cfg)

        self.nreg_moms = None


    def get_nregs_moms(self, nreg_moms):
        if self.nreg_moms != None and self.nreg_moms!=nreg_moms: 
            raise RuntimeError("Some issue")

        if self.nreg_moms == nreg_moms: return list(range(nreg_moms))

        # Moment storage
        nalph = self.sm.nalph
        eles_scal_upts_inb_moms = proxylist(self.ele_banks)

        # loop over the sub-domains in the full mixed domain
        for t, (nupts, nvars, neles) in enumerate(self.ele_shapes):
            eles_scal_upts_inb_moms[t] = self.backend.matrix_bank([
                    self.backend.matrix((nupts*neles, nalph)) 
                for i in range(nreg_moms)]
            )

        self.eles_scal_upts_inb_moms = eles_scal_upts_inb_moms
        self.nreg_moms = nreg_moms
        self.nalph = nalph
        return list(range(nreg_moms))


    # a utility function for extracting element-type data
    def ptr(self, bank, *vs):
        return [bank[v] if (isinstance(v, int)) else v for v in vs]


    def collide(self, *args, **kwargs):
        raise RuntimeError("Not valid for asymptotic systems")


    def moment(self, t, finbank, momoutbank, fscratchbank):
        fsoln = self.eles_scal_upts_inb_full
        moms = self.eles_scal_upts_inb_moms

        for t, ele in enumerate(self.ele_types):
            self.sm.swap_axes(fsoln[t][finbank], fsoln[t][fscratchbank]);
            self.sm.moment(t, fsoln[t][fscratchbank], moms[t][momoutbank]);


    def constructMaxwellian(self, t, mominbank, Mcoeffbank, momscratchbank):
        fsoln = self.eles_scal_upts_inb_full
        moms = self.eles_scal_upts_inb_moms
        
        for t, ele in enumerate(self.ele_types):
            self.sm.constructMaxwellian(
                moms[t][mominbank], fsoln[t][Mcoeffbank], 
                moms[t][momscratchbank]);


    def updateMomentARS(self, dt, *args):
        moms = self.eles_scal_upts_inb_moms        
        for t, ele in enumerate(self.ele_types):
            self.sm.updateMomentARS(dt, *self.ptr(moms[t], *args))


    def updateDistARS(self, dt, *args):
        q = (len(args) - 2)//6
        assert len(args)==6*q+2, "Inconsistency in number of parameters"

        cf, m, d = args[:3*q], args[3*q:4*q+1], args[4*q+1:]
        moms = self.eles_scal_upts_inb_moms        
        fsoln = self.eles_scal_upts_inb_full        

        for t, ele in enumerate(self.ele_types):
            self.sm.updateDistARS(dt, *[
                *self.ptr(fsoln[t], *cf), *self.ptr(moms[t], *m), 
                *self.ptr(fsoln[t], *d)
            ])


    def updateMomentBDF(self, dt, *args):
        moms = self.eles_scal_upts_inb_moms        
        for t, ele in enumerate(self.ele_types):
            self.sm.updateMomentBDF(dt, *self.ptr(moms[t], *args))

    def updateDistBDF(self, dt, *args):
        moms = self.eles_scal_upts_inb_moms        
        fsoln = self.eles_scal_upts_inb_full        

        for t, ele in enumerate(self.ele_types):
            self.sm.updateDistBDF(dt, *[
                *self.ptr(fsoln[t], *args[:-2]), *self.ptr(moms[t], args[-2]), 
                *self.ptr(fsoln[t], args[-1])
            ])

    def copy_dist(self, rdest, rsrc):
        fsoln = self.eles_scal_upts_inb_full
        for t, ele in enumerate(self.ele_types):
            cuda.memcpy_dtod(fsoln[t][rdest].data, fsoln[t][rsrc].data, 
                fsoln[t][rsrc].nbytes)

    def copy_moms(self, rdest, rsrc):
        moms = self.eles_scal_upts_inb_moms
        for t, ele in enumerate(self.ele_types):
            cuda.memcpy_dtod(moms[t][rdest].data, moms[t][rsrc].data, 
                moms[t][rsrc].nbytes)

