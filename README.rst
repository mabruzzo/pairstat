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

The easiest way to install the package as a user is to invoke:

.. code-block:: shell-session

   $ python -m pip install -v pyvsf@git+https://github.com/mabruzzo/pyvsf

Alternatively, you can clone the repository and invoke the following from the repository's root directory.

.. code-block:: shell-session

   $ python -m pip install -v .

OpenMP is used automatically used (if the compiler supports it). To check if the package was compiled with OpenMP, you can invoke the following from the command-line (and check if the printed statement mentions OpenMP)

.. code-block:: shell-session

   $ python -m pyvsf

If you want to use OpenMP on macOS, you will need to install a C++ compiler that supports it. The default C++ compiler on macOS is an apple-specific version of clang++.

- The easiest way to get a different compiler is use homebrew to install a version of g++.

- Once you have an installed version of g++, (like g++-14), you should invoke either

  .. code-block:: shell-session

     $ CXX=g++-14 python -m pip install -v pyvsf@git+https://github.com/mabruzzo/pyvsf

  or

  .. code-block:: shell-session

     $ CXX=g++-14 python -m pip install -v .

- NOTE: on macOS simply typing ``g++`` aliases the default clang++ compiler (``g++-14`` invokes a different compiler)


*****
Usage
*****

The primary 2 functions offered by this package are: ``pyvsf.vsf_props`` and ``pyvsf.twopoint_correlation``.
The former computes the structure function (we typically use it for the velocity structure function, but it could work with any vector-quantity).
The latter computes the two-point correlation function.

Each function operates on pairs of points, where each point has an associated value and spatial location.
In more detail, each function is associated with a distinct "pairwise-operation" that computes a scalar quantity associated with a pair of points. [#f1]_
The pairs of points can be grouped into discrete bins based on the spatial separation between the points in a pair.
For each separation bin, these functions computes a set of statistics describing the distribution of "pairwise-operation" values computed from every pair of points in the bin.

When this package is compiled with OpenMP support, the function can be parallelized.

In the next few subsections, we discuss:
- how to specify points
- how to specify the separation bins
- the available statistics

Specifying the points
=====================

They functions support 2 primary operation-modes:

1. Consider a single collection of points.
   In this case, the functions compute the "auto" structure function and the two-point auto-correlation function.
   The positions are specified via the ``pos_a`` argument and the values at each point are provided with the ``val_a`` argument.
   The caller must explicitly pass ``None`` to the ``pos_b`` and ``val_b`` arguments.

2. Consider 2 separate collections of points.
   In this case, the function computes "cross" structure function and the "cross"-two-point cross-correlation function.
   Like before, the positions and values for each point in the first collection are provided with ``pos_a`` and ``val_a``.
   The positions and values for each point the other collection are specified with ``pos_b`` and ``val_b``.

In both cases, positions should be specified in a 2D array, with a shape ``(3,N)``, where ``N`` specifies the number of points and ``3`` specifies the number of dimensions.

.. note::

   For now, we require 3-dimensional positions.
   To use the functions with 2-dimensional or 1-dimensional positions, just set the values along the unused dimension to a constant value.

When using ``pyvsf.vsf_props``, the values specify vector quantities (usually velocity) that have the same number of dimensions as the position.
In this case, the shape of ``val_a`` must match ``pos_a.shape`` and (if applicable) the shape of ``val_b`` must match ``pos_b.shape``.

When using ``pyvsf.twopoint_correlation``, the values specify scalar quantities.
In this case, ``val_a``  should be a 1D array with a shape ``(pos_a.shape[1],)``.
When it isn't ``None``, ``val_b`` should be a 1D array with a shape ``(pos_b.shape[1],)``.

Specify the Separation Bins
===========================

*[ NEEDS TO BE ADDED ]*

Set by the ``dist_bin_edges`` kwarg

Available Statistics
====================

The statistics are specified via the ``"stat_kw_pairs"`` keyword argument.
This expects a list of 1 or more pairs of statistic-kwarg pairs.
(This is a little clunky right now).
For now, you should just specify the name of a single statistic unless we explicitly note that a combination is supported.

Supported statistics include:

.. list-table:: Available Statistics
   :widths: 15 15 30
   :header-rows: 1

   * - name
     - ``stat_kw_pairs`` example
     - Description
   * - ``"mean"``
     - ``[("mean", {})]`` 
     - Computes the number of pairs and the mean
   * - ``"variance"``
     - ``[("variances", {})]`` 
     - Computes the number of pairs, the mean, and the variance.
   * - ``"histogram"``
     - ``[("histogram", {"val_bin_edges" : [0.0, 1.0, 2.0]})]``
     - Tracks the number of value computed for each pair of bins based on the specified ``"val_bin_edges"`` kwarg.
       The result is effectively a 2D histogram (the other axis is set by ``dist_bin_edges``.
       Not currently supported by ``pyvsf.twopoint_correlation``
   * - ``"weightedmean"``
     - ``[("weightedmean", {})]`` 
     - Computes the total weight and the weighted mean.
       Not supported by ``pyvsf.twopoint_correlation``
   * - ``"weightedhistogram"``
     - ``[("weightedhistogram", {"val_bin_edges" : [0.0, 1.0, 2.0]})]``
     - Tracks the total weight for all pairs of values that lie in the specified ``"val_bin_edges"`` bins.
       The result is effectively a 2D histogram (the other axis is set by ``dist_bin_edges``.
       Not currently supported by ``pyvsf.twopoint_correlation``

At the moment, you can chain together:
- ``"mean"`` and ``"histogram"``
- ``"variance"`` and ``"histogram"``
- ``"weightedmean"`` and ``"wightedhistogram"``


*******
Details
*******

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

.. rubric:: Footnotes

.. [#f1] The "pairwise-operation" for ``vsf_props`` computes the magnitude of the difference between 2 vectors. 
         For ``twopoint_correlation``, the "pairwise-operation" takes the product of 2 scalars.
