***************
Developer Guide
***************

Overview
========

This page documents some areas where the package needs work (contributions are welcomed)

There are 2 main points worth highlighting:

1. This module evolved very organically (features were added as they were needed). 
   A fair amount of refactoring could be done to simplify/improve certain aspects.
   Some of the required refactoring is described `here <https://github.com/mabruzzo/pyvsf/issues/1>`__.

2. There are a handful of oportunities for optimizing the performance of the package (most of them require significant structural changes).
   A description of som of these opportunities can be found `here <https://github.com/mabruzzo/pyvsf/issues/2>`__.


Undocumented Functionality
==========================

This module also provides another primary function, ``pyvsf.small_dist_sf_props.small_dist_sf_props`` that can be used to compute statistics for an astrophysical simulation.
This function decomposes the simulation into smaller subvolumes (the size of each subvolume is related to the maximum separation).
This can considerably reduce the complexity of the calculation.

``pyvsf.small_dist_sf_props.small_dist_sf_props`` also offers parallelization
using MPI/multiprocessing, using ``MPIPool`` or ``MultiPool`` from the ``schwimmbad`` package.
A modified `MPIPool` is also provided to work around some MPI issues on some computing clusters.

The plan is to move most of this undocumented functionality to a separate development branch prior to a 1.0 release.
