#!/usr/bin/env python3
# Author:  Name name
# Contact: bsc.mail@bsc.es

# -*- coding: utf-8 -*-
''' **ucis4eq** set up.
'''

# Imports
from setuptools.command.install import install


def name():
    ''' Set name for ucis4eq package.

    :param: None
    :return: ucis4eq name
    :rtype: string
    '''
    return 'ucis4eq'


def description():
    ''' Descripton of ucis4eq code
    '''
    ucis4eq_description = ('')

    return ucis4eq_description


def long_description():
    ''' Read long description of ucis4eq of DESCRIPTION.rst file.

    :param: None
    :return: ucis4eq description
    :rtype: string
    '''
    with open(os.path.join('DESCRIPTION.rst')) as f:
        return f.read()


def get_ext_modules():
    ''' Get paths of extension modules.

    :param: None
    :return: numpy path include
    '''
    try:
        import numpy
        numpy_includes = [numpy.get_include()]
    except ImportError:
        numpy_includes = []

    return numpy_includes


if __name__ == '__main__':
    from setuptools import setup
    import os

    # ucis4eq setup
    setup(name=name(),
          maintainer="juan.rodriguez",
          maintainer_email="juan.rodriguez@bsc.es",
          version='VERSION',
          long_description=long_description(),
          description=description(),
          url="https://www.bsc.es/",
          download_url="",
          license='To be defined',
          packages=['ucis4eq'],
          include_dirs=get_ext_modules(),
          install_requires=['numpy', 'flask', 'pymongo', 'requests', 'obspy'],
          setup_requires=['sphinx'],
          classifiers=['Development Status :: 3 - Alpha',
                       'License :: Free for non-commercial use',
                       'Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'Programming Language :: Python :: 3.5',
                       'Topic :: Scientific/Engineering',
                       'Topic :: Software Development :: Libraries',
                       'Operating System :: POSIX :: Linux'],
          keywords=['HPC, Hererogeneous Computing, Geophysics'],
          platforms="Linux",
          include_package_data=True
          )

else:
    pass
