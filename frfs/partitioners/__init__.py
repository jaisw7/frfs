# -*- coding: utf-8 -*-

from frfs.partitioners.base import BasePartitioner
from frfs.partitioners.metis import METISPartitioner
from frfs.util import subclass_where


def get_partitioner(name, *args, **kwargs):
    return subclass_where(BasePartitioner, name=name)(*args, **kwargs)
