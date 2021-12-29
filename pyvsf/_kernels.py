import numpy as np

"""
The basic idea here is to come up with a set of functions for each calculatable
type of Structure Function object that can be used to abstract over 
consolidation and such...
"""

from ._kernels_cy import (
    Variance,
    _validate_basic_quan_props,
    _allocate_unintialized_rslt_dict
)

class Mean:
    name = "mean"
    output_keys = ('counts', 'mean')
    commutative_consolidate = False
    operate_on_pairs = True
    non_vsf_func = None

    @classmethod
    def get_extra_fields(cls, kwargs = {}):
        return None

    @classmethod
    def get_dset_props(cls, dist_bin_edges, kwargs = {}):
        assert kwargs == {}
        assert np.size(dist_bin_edges) and np.ndim(dist_bin_edges) == 1
        return [('counts',   np.int64,   (np.size(dist_bin_edges) - 1,)),
                ('mean',     np.float64, (np.size(dist_bin_edges) - 1,))]

    @classmethod
    def consolidate_stats(cls, *rslts):
        # we could probably co-opt the method for the variance kernel
        raise NotImplementedError()

    @classmethod
    def validate_rslt(cls, rslt, dist_bin_edges, kwargs = {}):
        _validate_basic_quan_props(cls, rslt, dist_bin_edges, kwargs)

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, kwargs = {}):
        raise NotImplementedError()


class Histogram:
    name = "histogram"
    output_keys = ('2D_counts',)
    commutative_consolidate = True
    operate_on_pairs = True
    non_vsf_func = None

    @classmethod
    def get_extra_fields(cls, kwargs = {}):
        return None

    @classmethod
    def consolidate_stats(cls, *rslts):
        out = {}
        for rslt in rslts:
            if len(rslt) == 0:
                continue
            assert list(rslt.keys()) == ['2D_counts']

            if len(out) == 0:
                out['2D_counts'] = rslt['2D_counts'].copy()
            else:
                out['2D_counts'] += rslt['2D_counts']
        return out

    @classmethod
    def get_dset_props(cls, dist_bin_edges, kwargs = {}):
        assert list(kwargs.keys()) == ['val_bin_edges']
        val_bin_edges = kwargs['val_bin_edges']
        assert np.size(val_bin_edges) >= 2 and np.ndim(val_bin_edges) == 1
        assert np.size(dist_bin_edges) >= 2 and np.ndim(dist_bin_edges) == 1
        return [('2D_counts', np.int64,
                 (np.size(dist_bin_edges) - 1, np.size(val_bin_edges) - 1))]

    @classmethod
    def validate_rslt(cls, rslt, dist_bin_edges, kwargs = {},
                      used_points = None):
        _validate_basic_quan_props(cls, rslt, dist_bin_edges, kwargs)

        # do some extra validation
        hist_counts = rslt['2D_counts']
        if (hist_counts < 0).any():
            print('n_points', n_points)
            raise ValueError("The histogram can't contain negative counts")

        if used_points is not None:
            # compute the maximum number of pairs of points (make sure to
            # to compute this with python integers (to avoid overflows)
            max_pairs = int(n_points)*max(int(n_points-1),0)//2

            if max_pairs > np.iinfo(hist_counts.dtype).max:
                n_pairs = sum(int(e) for e in hist_counts)
            else:
                n_pairs = np.sum(hist_counts)
            if n_pairs > max_pairs:
                raise ValueError(
                    f"The dataset made use of {use_points} points. The "
                    f"histogram should hold no more than {max_pairs} pairs of "
                    f"points. In reality, it has {n_pairs} pairs."
                )
    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, kwargs = {}):
        # basically create a result object for a dataset that didn't have any
        # pairs at all
        rslt = _allocate_unintialized_rslt_dict(cls, dist_bin_edges, kwargs)
        for k in rslt.keys():
            rslt[k][...] = 0
        return rslt

def _compute_bulkaverage(quan, extra_quantities, kwargs):
    """
    Parameters
    ----------
    quan: np.ndarray
        Expected to be a (3,N) array of doubles that nominally hold N velocity
        values. This is not a unyt array.
    extra_quan: dict
        Dictionary where keys correspond to field names and values correspond
        to 1D arrays holding N values of that array (N should match 
        quan.shape[N]). This must contain 
    kwargs: dict
        This should be a 1-element dict. The key should be 'weight_field' and 
        the value should be a tuple, where the first element specifies the name
        of the weight field and the second element specifies the expected units.

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
    non_vsf_func = _compute_bulkaverage

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
        

_KERNELS = (Mean, Variance, Histogram, BulkAverage)
_KERNEL_DICT = dict((kernel.name, kernel) for kernel in _KERNELS)

def get_kernel(statistic):
    try:
        return _KERNEL_DICT[statistic]
    except KeyError:
        # the `from None` clause avoids exception chaining
        raise ValueError(f"Unknown Statistic: {statistic}") from None

def get_kernel_quan_props(statistic, dist_bin_edges, kwargs = {}):
    kernel = get_kernel(statistic)
    return kernel.get_dset_props(dist_bin_edges, kwargs)

def kernel_operates_on_pairs(statistic):
    return get_kernel(statistic).operate_on_pairs
