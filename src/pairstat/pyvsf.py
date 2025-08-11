# This only exists for backwards compatibility

from ._kernels_cy import vsf_props as vsf_props
import warnings

warnings.warn(
    "pairstat.pyvsf is deprecated. The submodule will be removed in the next release.",
    category=DeprecationWarning,
    stacklevel=2,
)
