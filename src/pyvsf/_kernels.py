from functools import partial
from types import new_class
import inspect

import numpy as np

"""
The basic idea here is to come up with a set of functions for each calculatable
type of Structure Function object that can be used to abstract over 
consolidation and such...
"""

from ._kernels_cy import _SF_STATCONF_TUPLE

from ._kernels_nonsf import BulkAverage, BulkVariance
from .grid_scale._kernels import GridscaleVdiffHistogram

# here we will do our best to dynamically generate the legacy Kernel types


class _MethodForwarder:
    def __init__(self, fn_name, fallback_fn, statconf_cls, forbid_fallback=False):
        self.fn_name = fn_name
        self.statconf_cls = statconf_cls
        self.signature = inspect.signature(fallback_fn)
        # we only forward to instance if kwargs is part of the signature
        self.forward_to_instance = "kwargs" in self.signature.parameters

        # now let's check if the statconf_cls has the method
        has_classmethod = inspect.ismethod(getattr(statconf_cls, fn_name, None))
        has_instancemethod = any(  # inspired by abstract base class subclass hook
            fn_name in klass.__dict__ for klass in statconf_cls.__mro__
        )
        if self.forward_to_instance and (has_classmethod or has_instancemethod):
            # it's ok if its a class method
            self.fallback_fn = None
        elif (not self.forward_to_instance) and has_classmethod:
            self.fallback_fn = None
        elif (not self.forward_to_instance) and has_instancemethod:
            raise ValueError(
                f"Can't forward to the {statconf_cls.__name__} StatConf class's "
                f"{fn_name} instance method. Can only forward to classmethod"
            )
        elif forbid_fallback:
            kind = ["class", "class or instance"][int(self.forward_to_instance)]
            raise ValueError(
                f"the {statconf_cls.__name__} StatConf class MUST provide a {kind} "
                f"method called {fn_name}"
            )
        else:
            self.fallback_fn = fallback_fn

    def __call__(self, cls, *args, **kwargs):
        as_kwargs = self.signature.bind(cls, *args, **kwargs).arguments
        del as_kwargs["cls"]
        if self.fallback_fn is not None:
            fn = self.fallback_fn
            return fn(cls, **as_kwargs)
        else:
            statconf_cls = self.statconf_cls

            if self.forward_to_instance:
                obj = statconf_cls(as_kwargs.pop("kwargs"))
                return getattr(obj, self.fn_name)(obj, **as_kwargs)
            else:
                # we don't need to pass statconf_cls as an argument
                return getattr(statconf_cls, self.fn_name)(**as_kwargs)


def _default_n_ghost_ax_end(cls):
    return 0


def _default_get_extra_fields(cls, kwargs=None):
    return None


def _default_consolidate_stats(cls, *rslts):
    raise RuntimeError("THIS SHOULD NOT BE CALLED ANY SF KERNEL")


def _stub_get_dset_props(cls, dist_bin_edges, kwargs=None):
    pass


def _stub_validate_rslt(cls, rslt, dist_bin_edges, kwargs=None):
    pass


def _stub_postprocess_rslt(cls, rslt):
    pass


def _default_zero_initialize_rslt(
    cls, dist_bin_edges, kwargs=None, postprocess_rslt=True
):
    raise NotImplementedError(f"has not been implemented yet")


# sequence of StatInfo names
def _make_kernel_class_callback(cls, statconf_cls):
    """
    The existing implementation of Kernels as collections of functions doesn't make a
    ton of sense any more. It has become apparent that it makes more sense for them to
    act like normal classes.

    To implement this gradually, the new StatConf classes act this way. Here we
    dynamically generate wrappers around the StatConf classes that provide the old
    kernel interface.
    """

    forwarders = (
        _MethodForwarder(
            "n_ghost_ax_end", _default_n_ghost_ax_end, statconf_cls, False
        ),
        _MethodForwarder(
            "get_extra_field", _default_get_extra_fields, statconf_cls, False
        ),
        _MethodForwarder("get_dset_props", _stub_get_dset_props, statconf_cls, True),
        _MethodForwarder(
            "consolidate_stats", _default_consolidate_stats, statconf_cls, False
        ),
        _MethodForwarder("validate_rslt", _stub_validate_rslt, statconf_cls, True),
        _MethodForwarder(
            "postprocess_rslt", _stub_postprocess_rslt, statconf_cls, True
        ),
        _MethodForwarder(
            "zero_initialize_rslt", _default_zero_initialize_rslt, statconf_cls, False
        ),
    )
    for forwarder_obj in forwarders:
        setattr(cls, forwarder_obj.fn_name, classmethod(forwarder_obj))

    # now let's deal with the class variable
    cls.name = statconf_cls.name
    cls.output_keys = statconf_cls.output_keys
    cls.commutative_consolidate = statconf_cls.commutative_consolidate
    cls.operate_on_pairs = getattr(statconf_cls, "operate_on_pairs", True)
    cls.requires_weights = statconf_cls.requires_weights
    cls.non_vsf_func = getattr(statconf_cls, "non_vsf_func", None)


def _make_kernel_classes():
    out = []
    for statconf_cls in _SF_STATCONF_TUPLE:
        callback = partial(_make_kernel_class_callback, statconf_cls=statconf_cls)
        name = "SFKernel" + statconf_cls.__name__
        out.append(new_class(name))
        callback(out[-1])
    return out


_SF_KERNEL_TUPLE = tuple(_make_kernel_classes())


class KernelRegistry:
    def __init__(self, itr):
        self._kdict = dict((kernel.name, kernel) for kernel in set(itr))

    def get_kernel(self, statistic):
        try:
            return self._kdict[statistic]
        except KeyError:
            # the `from None` clause avoids exception chaining
            raise ValueError(f"Unknown Statistic: {statistic}") from None


# sequence of kernels related to the structure function
_SF_KERNEL_REGISTRY = KernelRegistry(_SF_KERNEL_TUPLE)


_KERNELS = _SF_KERNEL_TUPLE + (BulkAverage, BulkVariance, GridscaleVdiffHistogram)
_KERNEL_REGISTRY = KernelRegistry(_KERNELS)


def get_kernel(statistic):
    return _KERNEL_REGISTRY.get_kernel(statistic)


def get_kernel_quan_props(statistic, dist_bin_edges, kwargs={}):
    kernel = get_kernel(statistic)
    return kernel.get_dset_props(dist_bin_edges, kwargs)


def kernel_operates_on_pairs(statistic):
    return get_kernel(statistic).operate_on_pairs
