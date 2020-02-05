# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np

from frfs.mpiutil import get_comm_rank_root, get_mpi
from frfs.plugins.base import BasePlugin, init_csv
from frfs.plugins.dgfsmomwriterstd import DGFSMomWriterStdPlugin
from frfs.writers.native import NativeWriter

class DGFSForceStdPlugin(BasePlugin):
    name = 'dgfsforcestd'
    systems = ['dgfs', 'adgfs']
    formulations = ['std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Output frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Boundary to integrate over
        bc = 'bcon_{0}_p{1}'.format(suffix, intg.rallocs.prank)
        self.suffix = suffix

        # Get the mesh and elements
        mesh, elemap = intg.system.mesh, intg.system.ele_map

        # See which ranks have the boundary
        bcranks = comm.gather(bc in mesh, root=root)

        # Output field names aka properties to be computed
        # px, py, pz: x, y, z components of normal pressure
        # fx, fy, fz: x, y, z components of force
        # q: total normal heat flux $\int Q_j n_j dA / \int dA$
        self.fields = (['fx', 'fy', 'fz'][:self.ndims] + ['q'])

        # create an instance of DGFSMomWriterStd class
        # Ceveat: One instance for every boundary :P
        self._moms = DGFSMomWriterStdPlugin(intg, cfgsect, suffix=None, 
            write=False)

        # The root rank needs to open the output file
        if rank == root:
            if not any(bcranks):
                raise RuntimeError('Boundary {0} does not exist'
                                   .format(suffix))

            # CSV header
            header = ['t'] + self.fields

            # Open
            self.outf = init_csv(self.cfg, cfgsect, ','.join(header))

        """
        # We need to dump the surface properties in a file
        # Construct the solution writer
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')
        
        # Output field names 
        self.fields = (['x', 'y', 'z'][:self.ndims] 
                        + ['nx', 'ny', 'nz'][:self.ndims] + self.fields)
        self._nvars = len(self.fields)
        self._writer = NativeWriter(intg, self._nvars, basedir, basename,
                                    prefix='soln')
        """

        # Interpolation matrices and quadrature weights
        self._m0 = m0 = {}
        self._qwts = qwts = defaultdict(list)

        # If we have the boundary then process the interface
        if bc in mesh:
            # Element indices and associated face normals
            eidxs = defaultdict(list)
            norms = defaultdict(list)
            mnorms = defaultdict(list)
            plocs = defaultdict(list)
            fidcount = dict()

            for etype, eidx, fidx, flags in mesh[bc].astype('U4,i4,i1,i1'):
                eles = elemap[etype]

                if (etype, fidx) not in m0:
                    facefpts = eles.basis.facefpts[fidx]

                    m0[etype, fidx] = eles.basis.m0[facefpts]
                    qwts[etype, fidx] = eles.basis.fpts_wts[facefpts]

                # Unit physical normals and their magnitudes (including |J|)
                npn = eles.get_norm_pnorms(eidx, fidx)
                mpn = eles.get_mag_pnorms(eidx, fidx)
                ploc = eles.get_ploc(eidx, fidx)

                eidxs[etype, fidx].append(eidx)
                norms[etype, fidx].append(mpn[:, None]*npn)
                mnorms[etype, fidx].append(mpn[:, None])
                plocs[etype, fidx].append(ploc[:,None])

                if etype not in fidcount:
                    fidcount[etype] = m0[etype, fidx].shape[0]
                else:
                    fidcount[etype] += m0[etype, fidx].shape[0]

            self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
            self._norms = {k: np.array(v) for k, v in norms.items()}
            self._mnorms = {k: np.array(v) for k, v in mnorms.items()}
            self._plocs = {k: np.array(v) for k, v in plocs.items()}

        # allocate variables
        #self._surfdata = [np.empty((fidcount[etype], self._nvars))
        #                    for etype in intg.system.ele_types]

        self(intg)

    def __call__(self, intg):
        # Return if no output is due
        if intg.nacptsteps % self.nsteps:
            return

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # compute moments
        self._moms.compute_moments(intg)
        soln = self._moms.bulksoln

        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, soln))
        ndims, nvars = self.ndims, soln[0].shape[1]

        # surf data
        #surfdata = dict(zip(intg.system.ele_types, self._surfdata))
        #cnt = dict()

        # Force vector
        f = np.zeros(ndims+2)
        for etc, (etype, fidx) in enumerate(self._m0):
            #if etc not in cnt: cnt[etc] = 0

            # Get the interpolation operator
            m0 = self._m0[etype, fidx]
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][..., self._eidxs[etype, fidx]]

            # Interpolate to the face
            ufpts = np.dot(m0, uupts.reshape(nupts, -1))
            ufpts = ufpts.reshape(nfpts, nvars, -1)
            ufpts = ufpts.swapaxes(0, 1)

            # Compute the pressure
            pidx = -1 # the pressure is the last variable (see privarmap)
            p = self.elementscls.con_to_pri(ufpts, self.cfg)[pidx]

            # compute the heat flux
            qidx = 5 if self.ndims==2 else 6
            q = [self.elementscls.con_to_pri(ufpts, self.cfg)[qidx+idx]
                    for idx in range(self.ndims)] 

            # Get the quadrature weights and normal vectors
            qwts = self._qwts[etype, fidx]
            norms = self._norms[etype, fidx]
            mnorms = self._mnorms[etype, fidx]
            plocs = np.squeeze(self._plocs[etype, fidx])

            # Compute forces
            f[:ndims] += np.einsum('i...,ij,jik', qwts, p, norms)

            # Compute total heat transfer
            f[ndims] += np.einsum('i...,kij,jik', qwts, q, norms)

            # Compute total area
            f[ndims+1] += np.sum(mnorms)

            """
            # coordinates and surface normals
            surfnorm = np.einsum('i...,jik->jk', qwts, norms)
            surfploc = np.einsum('i...,jik->jk', qwts, plocs)

            # append data
            print(np.hstack((surfploc[:,:ndims], surfnorm[:,:ndims])).shape)
            print(surfploc[:,:ndims].shape)
            print("tada:", self._surfdata[etc].shape)
            print("etc:", etc, ", etype:", etype)
            self._surfdata[etc][cnt[etc]:cnt[etc]+nfpts, :2*ndims]=np.hstack(
                                (surfploc[:,:ndims], surfnorm[:,:ndims]))
            self._surfdata[etc][cnt[etc]:cnt[etc]+nfpts, 2*ndims:]=f[:ndims+1]
            cnt[etc] += nfpts
            """

        # Reduce and output if we're the root rank
        if rank != root:
            comm.Reduce(f, None, op=get_mpi('sum'), root=root)
        else:
            comm.Reduce(get_mpi('in_place'), f, op=get_mpi('sum'), root=root)

            # compute the force
            #f[:ndims] *= f[ndims+1]

            # compute the total heat transfer per unit area
            f[ndims] /= f[ndims+1]

            # Build the row
            row = [intg.tcurr] + f[:-1].tolist()

            # write to console
            print(self.name, self.suffix, ':', ' ,'.join(str(r) for r in row))
            
            # Write
            print(','.join(str(r) for r in row), file=self.outf)

            # Flush to disk
            self.outf.flush()

        """
        # the surface data file
        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        stats.set('data', 'prefix', 'forces')
        intg.collect_stats(stats)

        # Prepare the metadata
        metadata = dict(intg.cfgmeta,
                        stats=stats.tostr(),
                        mesh_uuid=intg.mesh_uuid)

        # Write out the file
        solnfname = self._writer.write(self._surfdata, metadata, intg.tcurr)

        # If a post-action has been registered then invoke it
        self._invoke_postaction(mesh=intg.system.mesh.fname, soln=solnfname,
                                t=intg.tcurr)
        """

