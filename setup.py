import os
from setuptools import setup, find_packages
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

_EXTRA_COMPILE_ARG = '--std=c++17'

_PYVSF_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
_PYVSF_CPP_SRC_DIR = os.path.join(_PYVSF_ROOT_DIR,'src')

def _kernel_extension_module():
    include_dirs = [_PYVSF_CPP_SRC_DIR],
    if False:
        # this was the old strategy, here we relied on compiling the c++ files
        # into a shared library called libvsf.so before compiling the
        # _kernels_cy extension-module and then linking the extension module
        # against the precompiled shared library
        # -> unfortunately, we ran into all sorts of problems when compiling or
        #    running on macOS (basically the runtime_library_dirs did not seem
        #    to work properly)
        # -> there's definitely some ways around the issues, but this did not
        #    seem like the most portable solution...
        print(f'Path to dir containing shared library: "{_PYVSF_CPP_SRC_DIR}"')
        extra_sources = []
        library_dirs = [_PYVSF_CPP_SRC_DIR]
        runtime_library_dirs = [_PYVSF_CPP_SRC_DIR]
        libraries = ['vsf']
        extra_compile_args = [_EXTRA_COMPILE_ARG]
        extra_link_args = []

    else:
        # In this approach we compile the c++ files directly into the extension
        # module. Ideally, we would take some steps to automatically keep the
        # build options that are used here, synchronized with options used in
        # the Makefile
        extra_sources = ['./src/accum_handle.cpp', './src/vsf.cpp']
        library_dirs, runtime_library_dirs = [], []
        libraries = ['m']
        extra_compile_args = [_EXTRA_COMPILE_ARG, '-fno-math-errno']
        extra_link_args = []

        # this is a hack to avoid using openmp on the default apple-clang
        # compiler. In reality, we probably want to do something a little more
        # similar to yt and use a test-program to check if -fopenmp is
        # supported by the compiler (after all we could always use a different
        # compiler on a Macbook)
        import platform
        if platform.system() != 'Darwin':
            extra_compile_args.append('-fopenmp')
            extra_link_args.append('-fopenmp')

    return Extension('pyvsf._kernels_cy',
                     sources = ['pyvsf/_kernels_cy.pyx'] + extra_sources,
                     include_dirs = [_PYVSF_CPP_SRC_DIR],
                     library_dirs = library_dirs,
                     runtime_library_dirs = runtime_library_dirs,
                     libraries = libraries,
                     extra_compile_args = extra_compile_args,
                     extra_link_args = extra_link_args,
                     language="c++")
    
_kernels_Extension = None

ext_modules = [
    Extension('pyvsf._ArrayDict_cy', ['pyvsf/_ArrayDict_cy.pyx'],
              extra_compile_args = [_EXTRA_COMPILE_ARG],
              language="c++"),
    _kernel_extension_module(),
    Extension('pyvsf._partition_cy', ['pyvsf/_partition_cy.pyx'],
              include_dirs = [_PYVSF_CPP_SRC_DIR],
              extra_compile_args = [_EXTRA_COMPILE_ARG],
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
                      'yt', "schwimmbad", "pydantic"],
    author='Matthew Abruzzo',
    author_email='matthewabruzzo@gmail.com',
    packages=find_packages(exclude = ['tests', 'src', 'examples']),
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
