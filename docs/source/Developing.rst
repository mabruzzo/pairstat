***************
Developer Guide
***************

This page documents some areas where the package needs work.

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
using MPI/multiprocessing, using ``MPIPool`` or ``MultiPool`` from the ``schwimmbad`` package.
A modified `MPIPool` is also provided to work around some MPI issues on some computing clusters.

Need for Refactoring
====================
This module evolved very organically (features were added as they were needed). 
A fair amount of refactoring could be done to simplify/improve certain aspects.


