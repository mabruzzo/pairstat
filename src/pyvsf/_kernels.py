import numpy as np

"""
The basic idea here is to come up with a set of functions for each calculatable
type of Structure Function object that can be used to abstract over 
consolidation and such...
"""

from ._kernels_cy import _SF_KERNEL_TUPLE, KernelRegistry

from ._kernels_nonsf import BulkAverage, BulkVariance
from .grid_scale._kernels import GridscaleVdiffHistogram

_KERNELS = _SF_KERNEL_TUPLE + (BulkAverage, BulkVariance, GridscaleVdiffHistogram)
_KERNEL_REGISTRY = KernelRegistry(_KERNELS)


def get_kernel(statistic):
    return _KERNEL_REGISTRY.get_kernel(statistic)


def get_kernel_quan_props(statistic, dist_bin_edges, kwargs={}):
    kernel = get_kernel(statistic)
    return kernel.get_dset_props(dist_bin_edges, kwargs)


def kernel_operates_on_pairs(statistic):
    return get_kernel(statistic).operate_on_pairs
