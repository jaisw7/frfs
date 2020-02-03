# -*- coding: utf-8 -*-

from frfs.readers.base import BaseReader, NodalMeshAssembler
from frfs.readers.cgns import CGNSReader
from frfs.readers.gmsh import GmshReader

from frfs.util import subclasses, subclass_where


def get_reader_by_name(name, *args, **kwargs):
    return subclass_where(BaseReader, name=name)(*args, **kwargs)


def get_reader_by_extn(extn, *args, **kwargs):
    reader_map = {ex: cls
                  for cls in subclasses(BaseReader)
                  for ex in cls.extn}

    return reader_map[extn](*args, **kwargs)
