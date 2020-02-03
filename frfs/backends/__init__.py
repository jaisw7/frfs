# -*- coding: utf-8 -*-

from frfs.backends.base import BaseBackend
from frfs.backends.cuda import CUDABackend
from frfs.util import subclass_where


def get_backend(name, cfg):
    return subclass_where(BaseBackend, name=name.lower())(cfg)
