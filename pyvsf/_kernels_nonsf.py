"""
Define some statistical kernels that are unrelated to the calculation of the
structure function
"""
from copy import deepcopy

import numpy as np

from ._kernels_cy import (
    _validate_basic_quan_props,
    _allocate_unintialized_rslt_dict
)

def _generic_kernel_handle_args(quan, extra_quantities, kwargs):
    """
    Helper function that factors out some features for use in the main 
    functions used to compute non-structure function statistics.
    """
    if len(kwargs) != 1:
        raise ValueError("kwargs should be a dictionary holding 1 element")
    elif len(kwargs['weight_field']) != 2:
        raise ValueError(
            "kwargs['weight_field'] should hold the weight field's name and "
            "its expected units"
        )
    weight_field_name, weight_field_units = kwargs['weight_field']
    weights = extra_quantities[weight_field_name]
    if quan.size == 0:
        return {}
    elif np.ndim(quan) != 2:
        raise ValueError("quan must be a 2D array")
    assert quan.shape[0] == 3

    assert np.ndim(weights) == 1
    assert quan.shape[1] == weights.shape[0]
    return weights

def compute_bulkaverage(quan, extra_quantities, kwargs):
    """
    Parameters
    ----------
    quan: np.ndarray
        Expected to be a (3,N) array of doubles that nominally hold N velocity
        values. This is not a unyt array.
    extra_quan: dict
        Dictionary where keys correspond to field names and values correspond
        to 1D arrays holding N values of that array (N should match 
        quan.shape[1]).
    kwargs: dict
        This should be a 1-element dict. The key should be 'weight_field' and 
        the value should be a tuple, where the first element specifies the name
        of the weight field and the second element specifies the expected units.

    """
    weights = _generic_kernel_handle_args(quan, extra_quantities, kwargs)

    # axis = 1 seems counter-intuitive, but I've confirmed it's correct
    averages, sum_of_weights = np.average(quan, axis = 1,
                                          weights = weights,
                                          returned = True)

    # since sum_of_weights is 1D and has a length equal to quan.shape[1], all 3
    # entires are identical
    assert (sum_of_weights[0] == sum_of_weights).all()
    assert sum_of_weights[0] != 0.0 # we may want to revisit return vals if
                                    # untrue

    return {'average' : averages, 'weight_total' : sum_of_weights[:1]}

class BulkAverage:
    """
    This is used to directly compute weight average values for velocity 
    components.

    TODO: consider letting the number of components change
    TODO: consider handling multiple weight fields at once
    TODO: consider handling no weight field
    """
    name = "bulkaverage"
    output_keys = ('weight_total', 'average')
    commutative_consolidate = False
    operate_on_pairs = False
    non_vsf_func = compute_bulkaverage

    @classmethod
    def get_extra_fields(cls, kwargs = {}):
        assert len(kwargs) == 1
        assert len(kwargs['weight_field']) == 2
        weight_field_name, weight_field_units = kwargs['weight_field']
        return {weight_field_name : (weight_field_units, cls.operate_on_pairs)}

    @classmethod
    def get_dset_props(cls, dist_bin_edges, kwargs = {}):
        assert (len(kwargs) == 1) and ('weight_field' in kwargs)
        return [('weight_total',  np.float64, (1,)),
                ('average',       np.float64, (3,))]

    @classmethod
    def consolidate_stats(cls, *rslts):
        out = {}
        num_keys = len(cls.output_keys)

        # we could run into overflow problems.
        # We're using a compensated summation
        accum_prodsum = np.zeros((3,), np.float64)
        c_prodsum = np.zeros_like(accum_prodsum)    # needs to be zeros

        accum_wsum = np.zeros((1,), np.float64)
        c_wsum = np.zeros_like(accum_wsum)        # needs to be zeros

        first_filled_rslt = None
        for rslt in rslts:
            if len(rslt) == 0:
                continue
            if first_filled_rslt is None:
                first_filled_rslt = rslt

            assert len(rslt) == 2 # check the keys
            
            cur_weight = rslt['weight_total']
            cur_product = cur_weight*rslt['average']

            # first, accumulate weight
            cur_elem = cur_weight - c_wsum
            tmp_accum = accum_wsum + cur_elem
            c_wsum = (tmp_accum - accum_wsum) - cur_elem
            accum_wsum = tmp_accum

            # next, accumulate product
            cur_elem = cur_product - c_prodsum
            tmp_accum = accum_prodsum + cur_elem
            c_prodsum = (tmp_accum - accum_prodsum) - cur_elem
            accum_prodsum = tmp_accum

        if first_filled_rslt is None:
            return {}
        elif (accum_wsum == first_filled_rslt['weight_total']).all():
            return {'weight_total' : first_filled_rslt['weight_total'].copy(),
                    'average'      : first_filled_rslt['average'].copy()}
        else:
            weight_total, weight_times_avg  = accum_wsum, accum_prodsum
            return {'weight_total' : weight_total,
                    'average' : weight_times_avg / weight_total}

    @classmethod
    def validate_rslt(cls, rslt, dist_bin_edges, kwargs = {}):
        _validate_basic_quan_props(cls, rslt, dist_bin_edges, kwargs)

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, kwargs = {}):
        rslt = _allocate_unintialized_rslt_dict(cls, dist_bin_edges, kwargs)
        for k in rslt.keys():
            if k == 'average':
                rslt[k][:] = np.nan
            else:
                rslt[k][:] = 0.0
        return rslt

def _weighted_variance(x, axis = None, weights = None, returned = False):
    """
    Compute the weighted variance.

    We explicitly divide the sum of the squared deviations from the mean by the 

    Parameters
    ----------
    x: ndarray
        Array containing numbers whose weighted variance is desired
    axis: None or int
        Axis along which to average `values`
    weights: ndarray, optional
        An array of weights associated with the values in a. If `weights` is
        `None`, each value in `x` is assumed to have a weight of 1.
    returned: bool, optional
        Default is `False`. If True, the tuple 
        `(variance, average, sum_of_weights)` is returned, 
        otherwise only the variance is returned

    Notes
    -----
    If `x` and `weights` are 1D arrays, we explicitly use the formula:
        `var = np.sum( weights * np.square(x - mean) ) / np.sum(weight)`,
    where `mean = np.average(x, weights = weights)`.

    When the weights are all identically equal to 1, this is equivalent to:
        `var = np.sum( np.square(x - np.mean(x))**2 ) / x.size`

    To be clear, we are NOT trying to to return an unbiased estimate of the 
    variance. Further elaboration is given in the docstring for the 
    `BulkVariance` class.
    """
    

    # before anything else, compute the weighted average (if we're going to
    # need it)
    if returned or (weights is not None):
        mean, weight_sum = np.average(x, axis = axis, weights = weights,
                                      returned = True)

    if weights is None:
        variance = np.var(x, axis = axis, ddof = 0)
    elif (np.ndim(x) == 2) and (axis == 1) and (np.ndim(weights) == 1):
        x = np.asarray(x, dtype = np.float64)
        weights = np.asarray(weights, dtype = np.float64)
        assert x.shape[1:] == weights.shape
        variance = np.empty((x.shape[0],), dtype = np.float64)
        for i in range(variance.size):
            variance[i] = (np.sum(weights * np.square(x[i,:] - mean[i]))
                           / weight_sum[i])
    else:
        raise NotImplementedError("A generic implementation has not been "
                                  "provided")
    if returned:
        return variance, mean, weight_sum
    else:
        return variance

def compute_bulk_variance(quan, extra_quantities, kwargs):
    """
    Parameters
    ----------
    quan: np.ndarray
        Expected to be a (3,N) array of doubles that nominally hold N velocity
        values. This is not a unyt array.
    extra_quan: dict
        Dictionary where keys correspond to field names and values correspond
        to 1D arrays holding N values of that array (N should match 
        quan.shape[1]).
    kwargs: dict
        This should be a 1-element dict. The key should be 'weight_field' and 
        the value should be a tuple, where the first element specifies the name
        of the weight field and the second element specifies the expected units.

    """
    weights = _generic_kernel_handle_args(quan, extra_quantities, kwargs)

    # axis = 1 seems counter-intuitive, but I've confirmed it's correct
    variance, averages, sum_of_weights = _weighted_variance(quan, axis = 1,
                                                            weights = weights,
                                                            returned = True)

    # since sum_of_weights is 1D and has a length equal to quan.shape[1], all 3
    # entires are identical
    assert (sum_of_weights[0] == sum_of_weights).all()
    assert sum_of_weights[0] != 0.0 # we may want to revisit return vals if
                                    # untrue

    out = {'variance' : variance, 'average' : averages,
           'weight_total' : sum_of_weights[:1]}
    return out
    

class BulkVariance:
    """
    This is used to directly compute weight variance values for velocity 
    components.

    This uses the variance equation without the correction for bias. Reasons 
    are given below for why application of Bessel's correction doesn't make
    much sense. 

    Notes
    -----
    I've given this extensive thought, and I've decided it doesn't make any
    sense to try to apply a variant of Bessel's correction to this calculation
    (to try and get an unbiased estimate of the variance).

    There are three main points here that are worth consideration:
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
    
    Here are 2 compelling cases to consider:
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
    name = "bulkvariance"
    output_keys = ('weight_total', 'average', 'variance')
    commutative_consolidate = False
    operate_on_pairs = False
    non_vsf_func = compute_bulk_variance

    @classmethod
    def get_extra_fields(cls, kwargs = {}):
        assert len(kwargs) == 1
        assert len(kwargs['weight_field']) == 2
        weight_field_name, weight_field_units = kwargs['weight_field']
        return {weight_field_name : (weight_field_units, cls.operate_on_pairs)}

    @classmethod
    def get_dset_props(cls, dist_bin_edges, kwargs = {}):
        assert (len(kwargs) == 1) and ('weight_field' in kwargs)
        return [('weight_total',  np.float64, (1,)),
                ('average',       np.float64, (3,)),
                ('variance',      np.float64, (3,))]

    @classmethod
    def consolidate_stats(cls, *rslts):

        # find first non-empty result
        sample = next(filter(lambda e: e != {}, rslts), None)
        if sample is None:
            return {}
        else:
            # initialize output dict:
            out = dict((k, np.zeros_like(v)) for k,v in sample.items())

        # concatenate entries from each dict into 3 larger arrays
        n_rslts = len(rslts)
        if n_rslts == 1:
            return deepcopy(rslts[0])
        all_weight_totals = np.empty((n_rslts,) + sample['weight_total'].shape,
                                     dtype = np.float64)
        all_means = np.empty((n_rslts,) + sample['average'].shape,
                             dtype = np.float64)
        all_variances = np.empty((n_rslts,) + sample['variance'].shape,
                                 dtype = np.float64)

        assert all_weight_totals.shape == (n_rslts,1) # may change in future
        assert all_means.shape == all_variances.shape

        for i, rslt in enumerate(rslts):
            if rslt == {} or (rslt['weight_total'][0] == 0.0):
                all_weight_totals[i] = 0.0
                all_means[i] = 0.0
                all_variances[i] = 0.0
            else:
                all_weight_totals[i,0] = rslt['weight_total'][0]
                all_means[i,...] = rslt['average']
                all_variances[i,...] = rslt['variance']

        def func(local_weight_tots, local_means, local_vars):
            # this borrows heavily from the yt-project!
            dtype = np.float64

            global_weight_total = local_weight_tots.sum(dtype = dtype)
            if global_weight_total == 0.0:
                return np.nan, np.nan, 0.0

            global_mean = (
                (local_weight_tots*local_means).sum(dtype = dtype)
                / global_weight_total
            )

            delta2 = np.square(local_means - global_mean, dtype = dtype)
            global_var = (
                (local_weight_tots * (local_vars + delta2)).sum(dtype = dtype)
                / global_weight_total
            )

            return global_var, global_mean, global_weight_total

        assert all_variances.ndim == 2 # may change in the future
        for i in range(all_variances.shape[1]):
            global_var, global_mean, global_weight_total = func(
                all_weight_totals[:,0], all_means[:,i], all_variances[:,i]
            )
            if i == 0:
                out['weight_total'][0] = global_weight_total
            else:
                # sanity check:
                assert out['weight_total'][0] == global_weight_total
            out['average'][i] = global_mean
            out['variance'][i] = global_var

        if out['weight_total'][0] == 0.0:
            raise RuntimeError(
                "weight_total should never be 0. If it is, then we should be "
                "returning an empty dict instead"
            )

        return out


    @classmethod
    def validate_rslt(cls, rslt, dist_bin_edges, kwargs = {}):
        _validate_basic_quan_props(cls, rslt, dist_bin_edges, kwargs)

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, kwargs = {}):
        rslt = _allocate_unintialized_rslt_dict(cls, dist_bin_edges, kwargs)
        for k in rslt.keys():
            if k in ['variance', 'average']:
                rslt[k][:] = np.nan
            else:
                rslt[k][:] = 0.0
        return rslt
