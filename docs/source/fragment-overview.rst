********
Overview
********
This is a python package that implements C++ accelerated functions for computing the velocity structure function and the two-point correlation function for arbitrary sets of points.

These quantities are useful for characterizing the properties of turbulence.

The GitHub repository holding the source code can be found ``here <https://github.com/mabruzzo/pyvsf>``__ .

**********
Motivation
**********
Before developing this package, I used pure python functions that computed the velocity structure function and (and two-point correlation function) for arbitrary pairs of points by invoking the general-purpose ``scipy.spatial.distance.pdist`` and ``scipy.spatial.distance.cdist`` functions.
This package implements equivalent functionality that uses more specialized C++ in order to perform the calculation faster and with far less memory.

Crude benchmarking (see ``tests/vsf_props.py``) suggests that this package's functions are ~9 times faster for ~4e8 pairs (than the pure python equivalents)
For larger number of pairs of points, the performance gap may narrow to some extent, but this is regime where the pure python approach becomes untenable due to memory consumption.

Furthermore, when this package is compiled with OpenMP, it supports parallelization of these calculations.

