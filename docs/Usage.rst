*****
Usage
*****

The primary 2 functions offered by this package are: ``pairstat.vsf_props`` and ``pairstat.twopoint_correlation``.
The former computes the structure function (we typically use it for the velocity structure function, but it could work with any vector-quantity).
The latter computes the two-point correlation function.

Each function operates on pairs of points, where each point has an associated value and spatial location.
In more detail, each function is associated with a distinct "pairwise-operation" that computes a scalar quantity associated with a pair of points. [#f1]_
The pairs of points can be grouped into discrete bins based on the spatial separation between the points in a pair.
For each separation bin, these functions computes a set of statistics describing the distribution of "pairwise-operation" values computed from every pair of points in the bin.

When this package is compiled with OpenMP support, the function can be parallelized.

In the next few subsections, we discuss:

* how to specify points

* how to specify the separation bins

* the available statistics

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

When using ``pairstat.vsf_props``, the values specify vector quantities (usually velocity) that have the same number of dimensions as the position.
In this case, the shape of ``val_a`` must match ``pos_a.shape`` and (if applicable) the shape of ``val_b`` must match ``pos_b.shape``.

When using ``pairstat.twopoint_correlation``, the values specify scalar quantities.
In this case, ``val_a``  should be a 1D array with a shape ``(pos_a.shape[1],)``.
When it isn't ``None``, ``val_b`` should be a 1D array with a shape ``(pos_b.shape[1],)``.

.. note::

   For now, we require 3-dimensional positions.
   To use the functions with 2-dimensional or 1-dimensional positions, just set the values along the unused dimension to a constant value.

Specify the Separation Bins
===========================

Set by the ``dist_bin_edges`` kwarg.
This should monotonically increase and contain at least 2 elements.

Available Statistics
====================

The statistics are specified via the ``"stat_kw_pairs"`` keyword argument.
This expects a list of 1 or more pairs of statistic-kwarg pairs.
(This is a little clunky right now).
For now, you should just specify the name of a single statistic unless we explicitly note that a combination is supported.

Unweighted Statistics
---------------------

We provide a list of the unweighted statistics supported down below:

.. list-table:: Available Statistics
   :widths: 15 15 30
   :header-rows: 1

   * - name
     - ``stat_kw_pairs`` example
     - Description
   * - ``"mean"``
     - ``[("mean", {})]``
     - Computes the number of pairs and the mean.
       When used with :py:func:`~pairstat.vsf_props` function, the ``"mean"`` result correspond to the 1st order structure function.
       When used with :py:func:`~pairstat.twopoint_correlation` function, the ``"mean"`` result correspond is usually the quantity that you are interested in.
   * - ``"variance"``
     - ``[("variances", {})]``
     - Computes the number of pairs, the mean, and the variance.
       We currently apply Bessel's correction to try to get an unbiased estimate of variance.
   * - ``"omoment2"``
     - ``[("omoment2", {})]``
     - Computes the number of pairs, the mean, and the 2nd order moment about the origin.
       When used with :py:func:`pairstat.vsf_props` function, the ``"mean"`` and ``"omoment2"`` results correspond to the 1st and 2nd order structure functions.
   * - ``"omoment3"``
     - ``[("omoment3", {})]``
     - Computes the number of pairs and the mean.
       It also computes the 2nd and 3rd order moment about the origin.
       When used with :py:func:`pairstat.vsf_props` function, the ``"mean"``, ``"omoment2"``, and ``"omoment3"`` results correspond to the 1st, 2nd, and 3rd order structure functions.
   * - ``"omoment4"``
     - ``[("omoment4", {})]``
     - Computes the number of pairs and the mean.
       It also computes the 2nd and 3rd order moment about the origin.
       When used with :py:func:`pairstat.vsf_props` function, the ``"mean"``, ``"omoment2"``, ``"omoment3"``, and ``"omoment4"`` results correspond to the 1st, 2nd, 3rd, and 4th order structure functions.
   * - ``"histogram"``
     - ``[("histogram", {"val_bin_edges" : [0.0, 1.0, 2.0]})]``
     - Tracks the number of value computed for each pair of bins based on the specified ``"val_bin_edges"`` kwarg.

Weighted Statistics
-------------------

We also support weighted versions of each of the statistics described in the previous section.
To access these, you should prepend ``"weighted"`` to the start of the string (so ``"weightedmean"`` instead of ``"mean"`` or ``"weightedhistogram"`` instead of ``"histogram"``).
At the moment, these statistics can't be used with :py:func:`pairstat.twopoint_correlation`.

.. note::

   Unlike "variance", the "weightedvariance" statistic does **NOT** attempt to make any corrections to get an unbiased estimate of variance.


Chained Statistics
------------------

At the moment, you can chain together:

* ``"mean"`` and ``"histogram"``

* ``"variance"`` and ``"histogram"``

* ``"weightedmean"`` and ``"wightedhistogram"``

* ``"weightedvariance"`` and ``"wightedhistogram"``

.. rubric:: Footnotes

.. [#f1] The "pairwise-operation" for ``vsf_props`` computes the magnitude of the difference between 2 vectors.
         For ``twopoint_correlation``, the "pairwise-operation" takes the product of 2 scalars.
