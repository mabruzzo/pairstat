import numpy as np

"""
The basic idea here is to come up with a set of functions for each calculatable
type of Structure Function object that can be used to abstract over 
consolidation and such...
"""

from ._kernels_cy import (
    Histogram,
    Variance,
    _validate_basic_quan_props,
    _allocate_unintialized_rslt_dict,
    _set_empty_count_locs_to_NaN
)

from ._kernels_nonsf import BulkAverage, BulkVariance

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
        raise RuntimeError("THIS SHOULD NOT BE CALLED")

    @classmethod
    def validate_rslt(cls, rslt, dist_bin_edges, kwargs = {}):
        _validate_basic_quan_props(cls, rslt, dist_bin_edges, kwargs)

    @classmethod
    def postprocess_rslt(cls, rslt):
        _set_empty_count_locs_to_NaN(rslt)

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, kwargs = {},
                             postprocess_rslt = True):
        raise NotImplementedError()




_KERNELS = (Mean, Variance, Histogram, BulkAverage, BulkVariance)
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
