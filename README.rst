#####
pyvsf
#####


********
Overview
********
This is a python package that implements C++ accelerated functions for computing the velocity structure function and the two-point correlation function for arbitrary sets of points.

These quantities are useful for characterizing the properties of turbulence.

**********
Motivation
**********
Before developing this package, I used pure python functions that computed the velocity structure function and (and two-point correlation function) for arbitrary pairs of points by invoking the general-purpose ``scipy.spatial.distance.pdist`` and ``scipy.spatial.distance.cdist`` functions.
This package implements equivalent functionality that uses more specialized C++ in order to perform the calculation faster and with far less memory.

Crude benchmarking (see ``tests/vsf_props.py``) suggests that this package's functions are ~9 times faster for ~4e8 pairs (than the pure python equivalents)
For larger number of pairs of points, the performance gap may narrow to some extent, but this is regime where the pure python approach becomes untenable due to memory consumption.

Furthermore, when this package is compiled with OpenMP, it supports parallelization of these calculations.

************
Installation
************

This is fairly simple.
First, make sure you have a C++ compiler installed.

Then download a copy of the repository and from the root directory of the repository invoke

.. code-block:: shell-session

   $ python -m pip install -v .

OpenMP is used automatically used (if the compiler supports it).

If you want to use OpenMP on macOS, you will need to install a C++ compiler that supports it. The default C++ compiler on macOS is an apple-specific version of clang++.

- The easiest way to get a different compiler is use homebrew to install a version of g++.

- Once you have an installed version of g++, (like g++-14), you should invoke

  .. code-block:: shell-session

     $ CXX=g++-14 python -m pip install -v .

- NOTE: on macOS simply typing ``g++`` aliases the default clang++ compiler (``g++-14`` invokes a different compiler)

***********
Description
***********

The main function, ``pyvsf.vsf_props``, currently employs a naive
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

When this package is compiled with OpenMP support, the function can be parallelized.

***************
Developer Guide
***************

Optimization Opportunities
==========================

Faster algorithms, involving kdtrees/octrees, should definitely be
considered for larger problem sizes (the optimizations file briefly
talks about why these alternative approaches might be beneficial).

Another faster algorithm for regularly-spaced grid-based data would be
a stencil-based approach that allows you to determine the sparation
between pairs of points without actually calculating distances. An added
perk of this is that you can entirely remove the branching that is present
in the currently algorithm. As a consequence, vectorization would provide
a significant speed improvement.

Undocumented Functionality
==========================

This module also provides another primary function,
``pyvsf.small_dist_sf_props.small_dist_sf_props`` that can be used to
compute statistics for an astrophysical simulation. This function
decomposes the simulation into smaller subvolumes (the size of each
subvolume is related to the maximum separation). This can considerably
reduce the complexity of the calculation.

``pyvsf.small_dist_sf_props.small_dist_sf_props`` also offers parallelization
using MPI/multiprocessing, using ``MPIPool`` or ``MultiPool`` from the ``schwimmbad`` package. A modified `MPIPool` is also provided to work around some MPI issues
on some super computing clusters.

Need for Refactoring
====================
This module evolved very organically (features were added as they were needed). 
A fair amount of refactoring could be done to simplify/improve certain aspects.
