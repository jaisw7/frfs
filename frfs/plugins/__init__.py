# -*- coding: utf-8 -*-

from frfs.plugins.base import BasePlugin
from frfs.plugins.nancheck import NaNCheckPlugin
from frfs.plugins.dgfsmomwriterstd import DGFSMomWriterStdPlugin
from frfs.plugins.dgfsdistwriterstd import DGFSDistWriterStdPlugin
from frfs.plugins.dgfsresidualstd import DGFSResidualStdPlugin
from frfs.plugins.dgfsforcestd import DGFSForceStdPlugin
from frfs.plugins.dgfsmomwriterbi import DGFSMomWriterBiPlugin
from frfs.plugins.dgfsdistwriterbi import DGFSDistWriterBiPlugin
from frfs.plugins.dgfsresidualbi import DGFSResidualBiPlugin
from frfs.util import subclass_where


def get_plugin(name, *args, **kwargs):
    return subclass_where(BasePlugin, name=name)(*args, **kwargs)
