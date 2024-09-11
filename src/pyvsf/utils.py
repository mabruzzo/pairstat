# this is mostly for testing purposes

import numpy as np


def weighted_variance(x, axis=None, weights=None, returned=False):
    """
    Compute the weighted variance.

    We explicitly divide the sum of the squared deviations from the mean by
    the total weights. In other words we do NOT make any effort to apply a
    form of Bessel's correction. Reasoning is provided down below.

    Parameters
    ----------
    x: ndarray
        Array containing numbers whose weighted variance is desired
    axis: None or int
        Axis along which the variance is computed. The default behavior is
        to compute the variance of the flattened array.
    weights: ndarray, optional
        An array of weights associated with the values in a. If `weights` is
        `None`, each value in `x` is assumed to have a weight of 1.
    returned: bool, optional
        Default is `False`. If True, the tuple
        `(variance, average, sum_of_weights)` is returned,
        otherwise only the variance is returned

    Returns
    -------
    variance, [mean, weight_sum] : array_type or double
        Return the variance along the specified axis. When `returned` is
        `True`, this returns a tuple with the variance as the first
        element, the mean as the second element and sum of the weights as
        the third element.

    Notes
    -----
    If `x` and `weights` are 1D arrays, we explicitly use the formula:
        `var = np.sum( weights * np.square(x - mean) ) / np.sum(weight)`,
    where `mean = np.average(x, weights = weights)`.

    When the weights are all identically equal to 1, this is equivalent to:
        `var = np.sum( np.square(x - np.mean(x))**2 ) / x.size`

    To be clear, we are NOT trying to to return an unbiased estimate of the
    variance. Doing that involves the application of a variant of Bessel's
    correction. I've given this extensive thought, and I've decided it doesn't
    make any sense to try to do that.

    This was originally written under the pretext where we apply variance
    to some quantity in a numerical simulation and use some arbitrary
    weight field. There are three main points here that are worth
    consideration:
    - Bessel's correction for an unweighted variance has a smaller and smaller
      effect as the number of samples increase (the result gets closer and
      closer to biased estimator). In other words, it's correcting for the bias
      that arises from undersampling.

      - as an aside, Bessel's correction will give you a different result if
        you artificially inflated the number of samples. For example,
        if you duplicated every sample, the estimate of the mean and biased
        estimate of the variance remains unchanged. However, the unbiased
        variance estimate (with Bessel's correction) gives a different result.

    - The weight field doesn't need to be related to the number of samples in
      any way. It just so happens that samples in a unigrid simulation may have
      a fixed volume and samples in a particle-based simulation may have fixed
      mass. These are 2 special cases where one could get an unbiased
      volume-weighted and unbiased mass-weighted variance, respectively.

      - Therefore, in the general case, Bessel's correction does NOT get
        applied to the weight field

    - In this last point we consider 2 compelling scenarios:
      1. Suppose we wanted to measure a mass-weighted velocity average from a
         unigrid simulation. Consider these 2 alternative scenarios:
           a) We have 3 cells with masses of 4 g, 10g, and 100 g

           b) We have 7 cells with masses of 1 g, a cell with a mass of 8 g,
              and a cell with a mass of 97 g.

         For the sake of argument, let's assume mass is independent of velocity.
         If that's the case, then it's obvious that a smaller bias-correction
         is needed for the second case even though the first case has more
         mass (and one might argue that there's more 'stuff' in the first case)

           - it also becomes clear that there is no way to apply Bessel's
             correction based on the mass, since it's the choice of units would
             alter the correction.

       2. Consider a 2D AMR simulation, whose domain is subdivided into the
          following 5 blocks:

          .. code-block:: none

              +----+----+---------+
              | 1a | 1b |         |
              +----+----+    2    |
              | 1c | 1d |         |
              +----+----+---------+

         Suppose that each refined blocks has 0.25 the area of the coarse block
         and covers a quarter of the area. An area-weighted variance for some
         arbitrary quantity, using all of the cells in any one of these blocks
         would have the same level of bias-correction even though block 2
         covers a larger area.
         - there's probably *some* clever analytic formula that could be
           applied for correcting bias in area-weighted variances over the full
           domain. But, again that would be a special case. The same could
           not be said for weighting by some other quantity like mass
    """

    # before anything else, compute the weighted average (if we're going to
    # need it)
    if returned or (weights is not None):
        mean, weight_sum = np.average(x, axis=axis, weights=weights, returned=True)

    if weights is None:
        variance = np.var(x, axis=axis, ddof=0)
    elif axis is None:
        x = np.asarray(x).flatten()
        weights = np.asarray(weights).flatten()
        variance = np.sum(weights * np.square(x - mean)) / weight_sum
    elif (np.ndim(x) == 2) and (axis == 1) and (np.ndim(weights) == 1):
        x = np.asarray(x, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        assert x.shape[1:] == weights.shape
        variance = np.empty((x.shape[0],), dtype=np.float64)
        for i in range(variance.size):
            variance[i] = np.sum(weights * np.square(x[i, :] - mean[i])) / weight_sum[i]
    else:
        raise NotImplementedError("A generic implementation has not been provided")
    if returned:
        return variance, mean, weight_sum
    else:
        return variance
