########
pairstat
########


.. image:: https://github.com/mabruzzo/pairstat/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/mabruzzo/pairstat/actions
    :alt: pairstat's GitHub Actions CI Status

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://github.com/pre-commit/pre-commit
    :alt: pre-commit

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

`Documentation <https://pairstat.readthedocs.io/en/latest/>`__ |
`Installation <https://pairstat.readthedocs.io/en/latest/Install.html>`__ |
`Contributing <https://pairstat.readthedocs.io/en/latest/Contributing.html>`__ |
`Getting Help <https://pairstat.readthedocs.io/en/latest/Help.html>`__


.. COMMENT:  README-MAIN-BODY-START-ANCHOR

``pairstat`` is a python package that provides accelerated/parallelized routines for computing spatial 2-point statistics from spatial data data (e.g. 2-point correlation function, structure functions).

    The ``pairstat`` package was formerly known as ``pyvsf``

**********
Motivation
**********

2-point statistics are important for characterizing the properties of `turbulence <https://en.wikipedia.org/wiki/Turbulence#Kolmogorov's_theory_of_1941>`__ (2-point statistics comes up in other contexts like `cosmology <https://en.wikipedia.org/wiki/Correlation_function_(astronomy)>`__).
There hasn't been an easy-to-use package for computing these quantities, until now.

The ``pairstat`` package is most useful for datasets where Fourier methods are problematic (e.g. you don't have a regularly spaced periodic grid).
Before developing ``pairstat``, I performed similar calculations by processing the outputs of ``scipy.spatial.distance.pdist`` and ``scipy.spatial.distance.cdist`` functions.
This package implements equivalent functionality that uses more specialized C++ code in order to perform the calculation faster and with **far** less memory. [#of1]_
It also supports parallelization (more on that below).

************
Installation
************

As long as you have a C++ compiler, the easiest way to get the package is by invoking

.. code-block:: shell-session

   $ python -m pip install pairstat

The package is automatically compiled with OpenMP support if the compiler supports it.
To confirm that ``pairstat`` was compiled with OpenMP support, you can check whether the output from the following command mentions OpenMP:

.. code-block:: shell-session

   $ python -m pairstat

See our `Installation Guide <https://pairstat.readthedocs.io/en/latest/Install.html>`__ for more details (especially if the package wasn't compiled with OpenMP support).


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
We are planning to replace the C++ and Cython logic with the rust logic before the 1.0 release. This rewrite will allow us to significantly improve the code quality.

Contributions and Feature requests are welcome!

*******
License
*******
pairstat is dual-licensed under either the MIT license and the Apache License (Version 2.0).



.. rubric:: Footnotes

.. [#of1] Crude benchmarking (see ``tests/test_vsf_props.py``) suggests that this package's functions are ~9 times faster for ~4e8 pairs (than the pure python equivalents)
          For larger number of pairs of points, the performance gap may narrow to some extent, but this is regime where the pure python approach becomes untenable due to memory consumption.
