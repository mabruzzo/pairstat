# pyvsf

## Overview
This python module defines and wraps a simple C++ library that is used to
compute properties related to the velocity structure function.

This module is still highly experimental.

## Installation

This is fairly simple. From the root of the directory execute:

```console
$ python setup.py develop
```

In principle, other invocations may also work...

Right now, if you are on macOS, the code will not be compiled with openmp support, by default.
This is for compatibility with the default compiler shipped on macOS.

## Old Installation

To use the old installation method, set the first if-statement in the `_kernel_extension_module()` within `setup.py` to `True`.
This older approach runs into problems on macOS.

Installation is a little unorthodox since I have not had a chance to
figure out how to have the ``setup.py`` script compile the C++ library
itself. This requires that you have a C++ compiler (so far it has only
been tested with g++)

To install this module, you need to clone this repository and execute
the following commands from the root level of the repository:

```console
$ make
$ python setup.py develop
```

There are two important things to note:

- This currently needs to be installed in development mode (so that the module
  can find the shared library). This means that you can't delete this
  repository after installation

- You need to modify the ``Makefile`` if you have a C++ compiler other ``g++``

- You especially need modify the ``Makefile`` if you are using the Apple-provided clang-compiler on a Mac.
  This is especially important because the Apple-provided clang-compiler does NOT support openmp.
  The modification in this case is simple (instructions are provided telling you which 2 lines to comment and the other 2 lines that must be uncommented)

## Description

The main function, ``pyvsf.vsf_props``, currently employs a very naive
brute-force algorithm. The user specifies a set of distance bins and
either:

- the position and velocity properties for two sets of points.
- the position and velocity properties for a single set of points.

In the former case, the function considers all unique pairs between
the two sets of points while in the latter it considers just the
unique pairs in the single set of points.  For every pair of points
this function computes the distance between the points and identifies
the distance bin that this pair is a member of. The function returns
statistical properties (e.g. count, mean, variance) for the absolute
velocity differences in each bin.

Faster algorithms, involving kdtrees/octrees, should definitely be
considered for larger problem sizes (the optimizations file briefly
talks about why these alternative approaches might be beneficial).

Another faster algorithm for regularly-spaced grid-based data would be
a stencil-based approach that allows you to determine the sparation
between pairs of points without actually calculating distances. An added
perk of this is that you can entirely remove the branching that is present
in the currently algorithm. As a consequence, vectorization would provide
a significant speed improvement.

This module also provides another primary function,
``pyvsf.small_dist_sf_props.small_dist_sf_props`` that can be used to
compute statistics for an astrophysical simulation. This function
decomposes the simulation into smaller subvolumes (the size of each
subvolume is related to the maximum separation). This can considerably
reduce the complexity of the calculation.

## Parallelization

``pyvsf.vsf_props`` is currently parallelized for cross-structure functions
using OpenMP (most of the ground-work is there for auto-structure functions,
but that remains untested).

``pyvsf.small_dist_sf_props.small_dist_sf_props`` also offers parallelization
using MPI/multiprocessing, using `MPIPool` or `MultiPool` from the schwimmbad
package. A modified `MPIPool` is also provided to work around some MPI issues
on some super computing clusters.

## Motivations

The main motivation for this module was to have an alternative to using
``scipy.spatial.distance.pdist``/``scipy.spatial.distance.cdist`` with
numpy functions for computing velocity structure function that uses
considerably less memory. Crude benchmarking (see ``tests/vsf_props.py``)
suggests that this is ~9 times faster for ~4e8 pairs.

For larger numbers of pairs, it seems that the performance gap may narrow
somewhat. However this is precisely where the scipy/numpy approach becomes
untenable due to memory consumption.

## Caveats
This module evolved very organically. As a result, there are some oddities
(e.g. the use of Cython and the `ctypes` module). This is particularly true for
the ``pyvsf.small_dist_sf_props.small_dist_sf_props`` function. A fair amount
of refactoring could be done to simplify/improve certain aspects.