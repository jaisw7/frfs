# -*- coding: utf-8 -*-

from frfs.integrators import get_integrator
from frfs.solvers.base import BaseSystem
from frfs.solvers.dgfs import DGFSSystem
from frfs.solvers.dgfsbi import DGFSBiSystem
from frfs.util import subclass_where


def get_solver(backend, rallocs, mesh, initsoln, cfg):
    systemcls = subclass_where(BaseSystem, name=cfg.get('solver', 'system'))

    # Combine with an integrator to yield the solver
    return get_integrator(backend, systemcls, rallocs, mesh, initsoln, cfg)
