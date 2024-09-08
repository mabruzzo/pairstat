********
Overview
********
This is a python package that implements C++ accelerated functions for computing the structure function and the two-point correlation function for arbitrary sets of points.
These functions can be used for computing 1D, 2D, or 3D structure functions and correlation functions in numerical simulations and observations.
These quantities are useful for characterizing the properties of turbulence.

The GitHub repository holding the source code can be found `here <https://github.com/mabruzzo/pyvsf>`__ .

**********
Motivation
**********

Before developing this package, I performed similar calculations by processing the outputs of ``scipy.spatial.distance.pdist`` and ``scipy.spatial.distance.cdist`` functions.
This package implements equivalent functionality that uses more specialized C++ code in order to perform the calculation faster and with **far** less memory. [#of1]_ 
It also supports parallelization (more on that below).

*****************************************
Key-Features: Parallelism and Scalability
*****************************************

The key feature of this package is the support for parallelism.
If a compatible compiler is used to build this package, it will automatically be built with OpenMP support for parallelizing calculations of structure functions and correlation functions.

Undocumented machinery also exists to help use this functionality to parallelize calculations across machines on a computing cluster (e.g. with MPI).
We plan to document this machinery in the near future.

The other important feature, is memory usage.
The memory usage is independent of the number of points.
A naive implementation of equivalent calculation using scipy functionality has memory usage that scales with the number of pairs of points (i.e. the number of points squared for auto-correlation).
In other words, this function is far more scalable that the alternative.

**************
Current Status
**************
The functionality in this package was developed for a particular science project that involved computing velocity structure functions from uniform resolution Enzo-E simulations of cloud-wind interactions.
However, I later realized that the core functionality could be useful to a lot of people in a lot of contexts.

The core functionality has already been refactored and is presented in this documentation as part of public API.
I will do my best to maintain this compatibility with this API, but breaking changes may creep in before the 1.0 release.

There is still a large amount of undocumented machinery in this repository.
I plan to expose part of that machinery as part of the public API before the 1.0 release and remove the rest of it.

Contributions and Feature requests are welcome!

.. rubric:: Footnotes

.. [#of1] Crude benchmarking (see ``tests/vsf_props.py``) suggests that this package's functions are ~9 times faster for ~4e8 pairs (than the pure python equivalents)
          For larger number of pairs of points, the performance gap may narrow to some extent, but this is regime where the pure python approach becomes untenable due to memory consumption.


