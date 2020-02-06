#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from setuptools import setup
import sys


# Python version
if sys.version_info[:2] < (3, 3):
    print('PyFR requires Python 3.3 or newer')
    sys.exit(-1)

# PyFR version
vfile = open('frfs/_version.py').read()
vsrch = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", vfile, re.M)

if vsrch:
    version = vsrch.group(1)
else:
    print('Unable to find a version string in frfs/_version.py')

# Modules
modules = [
    'frfs.backends',
    'frfs.backends.base',
    'frfs.backends.cuda',
    'frfs.backends.cuda.kernels',
    'frfs.integrators',
    'frfs.integrators.std',
    'frfs.plugins',
    'frfs.quadrules',
    'frfs.readers',
    'frfs.partitioners',
    'frfs.solvers',
    'frfs.solvers.base',
    'frfs.writers',

    'frfs.sphericaldesign',

    'frfs.solvers.dgfs',
    'frfs.solvers.dgfs.kernels',
    'frfs.solvers.dgfs.kernels.bcs',
    'frfs.solvers.dgfs.kernels.rsolvers',
    'frfs.solvers.dgfs.kernels.scattering',

    'frfs.solvers.dgfsbi',
    'frfs.solvers.dgfsbi.kernels',
    'frfs.solvers.dgfsbi.kernels.bcs',
    'frfs.solvers.dgfsbi.kernels.rsolvers',
    'frfs.solvers.dgfsbi.kernels.scattering'

    'frfs.solvers.adgfs',
    'frfs.solvers.adgfs.kernels',
    'frfs.solvers.adgfs.kernels.scattering',
]

# Data
package_data = {
    'frfs.backends.cuda.kernels': ['*.mako'],
    'frfs.quadrules': [
        'hex/*.txt',
        'line/*.txt',
        'pri/*.txt',
        'pyr/*.txt',
        'quad/*.txt',
        'tet/*.txt',
        'tri/*.txt', 
        'point/*.txt'
    ],

    'frfs.sphericaldesign': [
        'symmetric/*.txt'
    ],

    'frfs.solvers.dgfs.kernels': ['*.mako'],
    'frfs.solvers.dgfs.kernels.bcs': ['*.mako'],
    'frfs.solvers.dgfs.kernels.rsolvers': ['*.mako'],
    'frfs.solvers.dgfs.kernels.scattering': ['*.mako'],

    'frfs.solvers.dgfsbi.kernels': ['*.mako'],
    'frfs.solvers.dgfsbi.kernels.bcs': ['*.mako'],
    'frfs.solvers.dgfsbi.kernels.rsolvers': ['*.mako'],
    'frfs.solvers.dgfsbi.kernels.scattering': ['*.mako']

    'frfs.solvers.adgfs.kernels': ['*.mako'],
    'frfs.solvers.adgfs.kernels.scattering': ['*.mako'],
}

# Additional data
data_files = [
    ('', ['frfs/__main__.py'])
]

# Hard dependencies
install_requires = [
    'appdirs >= 1.4.0',
    'gimmik >= 2.0',
    'h5py >= 2.6',
    'mako >= 1.0.0',
    'mpi4py >= 2.0',
    'numpy >= 1.8',
    'pytools >= 2016.2.1',
    'pycuda >= 2015.1'
]

# Soft dependencies
extras_require = {
}

# Scripts
console_scripts = [
    'frfs = frfs.__main__:main'
]

# Info
classifiers = [
    'License :: GNU GPL v2',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.3',
    'Topic :: Scientific/Engineering'
]

long_description = '''frfs is an open-source minimalistic implementation of 
Discontinuous Galerkin schemes for full Boltzmann equation and related kinetic
models via flux reconstruction.'''

setup(name='frfs',
      version=version,
      description='High order schemes for kinetic equations',
      long_description=long_description,
      author='Shashank Jaiswal',
      author_email='jaisw7@gmail.com',
      url='http://www.github.com/jaisw7',
      license='GNU GPL v2',
      keywords='Applied Mathematics; Kinetic Theory; High order schemes',
      packages=['frfs'] + modules,
      package_data=package_data,
      data_files=data_files,
      entry_points={'console_scripts': console_scripts},
      install_requires=install_requires,
      extras_require=extras_require,
      classifiers=classifiers
)
