# -*- coding: utf-8 -*-

import numpy as np

from frfs.mpiutil import get_comm_rank_root, get_mpi
from frfs.plugins.base import BasePlugin, init_csv


class DGFSResidualStdPlugin(BasePlugin):
    name = 'dgfsresidualstd'
    systems = ['dgfs', 'adgfs']
    formulations = ['std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Output frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')
        self.isoutf = self.cfg.getint(cfgsect, 'output-file', 0)

        # The root rank needs to open the output file
        if rank == root and self.isoutf:
            header = ['t', 'f']

            # Open
            self.outf = init_csv(self.cfg, cfgsect, ','.join(header))

        # Call ourself in case output is needed after the first step
        self(intg)

    def __call__(self, intg):
        # If an output is due this step
        if intg.nacptsteps % self.nsteps == 0 and intg.nacptsteps:
            # MPI info
            comm, rank, root = get_comm_rank_root()

            # Previous and current solution
            prev = self._prev
            curr = [s[intg._idxcurr].get() for s in 
                    intg.system.eles_scal_upts_inb_full]

            # Square of the residual vector [pad 0 for communication]
            resid_num = np.array([sum(np.linalg.norm(c - p)**2
                        for p, c in zip(prev, curr)), 0.])
            resid_den = np.array([sum(np.linalg.norm(p)**2
                        for p in prev), 0.])

            # Reduce and, if we are the root rank, output
            if rank != root:
                comm.Reduce(resid_num, None, op=get_mpi('sum'), root=root)
                comm.Reduce(resid_den, None, op=get_mpi('sum'), root=root)
            else:
                comm.Reduce(get_mpi('in_place'), resid_num, op=get_mpi('sum'),
                            root=root)
                comm.Reduce(get_mpi('in_place'), resid_den, op=get_mpi('sum'),
                            root=root)

                # Normalise [Remove the padded 0]
                resid = np.sqrt(resid_num[:-1]/resid_den[:-1])

                # Build the row
                row = [intg.tcurr] + resid.tolist()

                # Write
                print(' ', self.name, ': ', 
                    ', '.join("{0:.3e}".format(r) for r in row))

                # Flush to disk
                if(self.isoutf):
                    print(','.join(str(r) for r in row), file=self.outf)
                    self.outf.flush()

            del self._prev

        # If an output is due next step
        if (intg.nacptsteps + 1) % self.nsteps == 0:
            self._prev = [s[intg._idxcurr].get() for s in 
                            intg.system.eles_scal_upts_inb_full]
