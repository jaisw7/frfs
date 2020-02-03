# -*- coding: utf-8 -*-

from frfs.writers.base import BaseWriter
from frfs.writers.vtk import VTKWriter

from frfs.util import subclasses, subclass_where


def get_writer_by_name(name, *args, **kwargs):
    return subclass_where(BaseWriter, name=name)(*args, **kwargs)


def get_writer_by_extn(extn, *args, **kwargs):
    writer_map = {ex: cls
                  for cls in subclasses(BaseWriter)
                  for ex in cls.extn}

    return writer_map[extn](*args, **kwargs)
