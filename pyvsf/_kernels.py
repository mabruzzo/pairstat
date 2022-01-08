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
