import os
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    # subclass setuptools extension builder to avoid importing numpy
    # at top level in setup.py. See http://stackoverflow.com/a/21621689/1382869
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process
        # see http://stackoverflow.com/a/21621493/1382869
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

_PYVSF_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
_PYVSF_CPP_SRC_DIR = os.path.join(_PYVSF_ROOT_DIR,'src')
print(_PYVSF_CPP_SRC_DIR)

ext_modules = [
    Extension('pyvsf._ArrayDict_cy', ['pyvsf/_ArrayDict_cy.pyx'],
              language="c++"),
    Extension('pyvsf._kernels_cy', ['pyvsf/_kernels_cy.pyx'],
              include_dirs = [_PYVSF_CPP_SRC_DIR],
              library_dirs = [_PYVSF_CPP_SRC_DIR],
              runtime_library_dirs=[_PYVSF_CPP_SRC_DIR],
              libraries = ['vsf'],
              language="c++"),
]

# on some platforms, we need to apply the language level directive before setup
# (see https://stackoverflow.com/a/58116368)
for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(
    name='pyvsf',
    version='0.0.1',
    description='Module for computing velocity structure function properties',
    setup_requires = ['numpy', 'cython'],
    install_requires=['setuptools', 'numpy','h5py','cython', 'matplotlib',
                      'yt'],
    author='Matthew Abruzzo',
    author_email='matthewabruzzo@gmail.com',
    packages=['pyvsf'],
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
