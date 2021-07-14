# pyvsf

## Overview
This python module defines and wraps a simple C++ library that is used to
compute properties related to the velocity structure function.

This module is still highly experimental.

## Installation

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

## Motivations

The main motivation for this module was to have an alternative to using
``scipy.spatial.distance.pdist``/``scipy.spatial.distance.cdist`` with
numpy functions for computing velocity structure function that uses
considerably less memory. Crude benchmarking (see ``tests/vsf_props.py``)
suggests that this is ~9 times faster for ~4e8 pairs.

For larger numbers of pairs, it seems that the performance gap may narrow
somewhat. However this is precisely where the scipy/numpy approach becomes
untenable due to memory consumption.
