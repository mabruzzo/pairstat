# pyvsf

This python module defines and wraps a simple C++ library that is used to
compute properties related to the velocity structure function.

This module is still highly experimental.

The main function, ``pyvsf.vsf_props``, currently employs a very naive
algorithm. The user specifies the position and velocity properties for two sets
of points and a set of distance bins. For every unique pair of points between
the sets of points, this function computes the distance between the points
and identifies the distance bin that this pair is a member of. The function
returns statistical properties (e.g. count, mean, variance) for the absolute
velocity differences in each bin.

The main motivation for this module was to have an alternative to
``scipy.spatial.distance.pdist`` for computing velocity structure function that
uses considerably less memory.

Faster algorithms, involving kdtrees/octrees, should definitely be considered
for larger problem sizes.