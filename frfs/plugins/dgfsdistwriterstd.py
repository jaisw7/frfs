# -*- coding: utf-8 -*-

from frfs.inifile import Inifile
from frfs.plugins.base import BasePlugin
from frfs.writers.native import NativeWriter


class DGFSDistWriterStdPlugin(BasePlugin):
    name = 'dgfsdistwriterstd'
    systems = ['dgfs', 'adgfs']
    formulations = ['std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Construct the solution writer
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')

        # fix nvars: the full distribution
        self.nvars = intg.system.vm.vsize()

        self._writer = NativeWriter(intg, self.nvars, basedir, basename,
                                    prefix='soln')

        # Output time step and next output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_next = intg.tcurr

        # Output field names (Should be same as distvar in elementcls)
        self.fields = ['f_'+str(i) for i in range(self.nvars)]

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
        stats.set('data', 'prefix', 'soln')
        intg.collect_stats(stats)

        # Prepare the metadata
        metadata = dict(intg.cfgmeta,
                        stats=stats.tostr(),
                        mesh_uuid=intg.mesh_uuid)

        # Write out the file
        full_soln = [eb[intg._idxcurr].get() 
            for eb in intg.system.eles_scal_upts_inb_full]
        solnfname = self._writer.write(full_soln, metadata, intg.tcurr)

        # If a post-action has been registered then invoke it
        self._invoke_postaction(mesh=intg.system.mesh.fname, soln=solnfname,
                                t=intg.tcurr)

        # Compute the next output time
        self.tout_next = intg.tcurr + self.dt_out
