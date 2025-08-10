from collections.abc import Sequence
from typing import Callable, List, NamedTuple, Optional, Tuple, Type, Union
import warnings

import numpy as np

from libc.stdint cimport int64_t, uintptr_t
from libc.stddef cimport size_t

from cpython.version cimport PY_MAJOR_VERSION

from ._ArrayDict_cy import ArrayMap

def _verify_bin_edges(bin_edges):
    nbins = np.size(bin_edges) - 1
    if np.ndim(bin_edges) != 1:
        return False
    elif nbins <= 0:
        return False
    elif not np.greater(bin_edges[1:], np.array(bin_edges)[:-1]).all():
        return False
    else:
        return True

def _check_bin_edges_arg(arg, arg_description):
    if not _verify_bin_edges(arg):
        raise ValueError(f"The {arg_description} must specify a 1D array with "
                         "2 or more elements that monotonically increase")

def _check_dist_bin_edges(dist_bin_edges):
    _check_bin_edges_arg(dist_bin_edges, "'dist_bin_edges' argument")

cdef extern from "vsf.hpp":
    ctypedef struct PointProps:
        double* positions
        double* values
        double* weights
        size_t n_points
        size_t n_spatial_dims
        size_t spatial_dim_stride

    ctypedef struct BinSpecification:
        double* bin_edges
        size_t n_bins

    ctypedef struct ParallelSpec:
        size_t nproc
        bint force_sequential

    ctypedef struct StatListItem:
        char* statistic
        void* arg_ptr

    # at the end of the cython documentation page on using C++ in cython
    #     https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html
    # there is an discussion that cython generates calls to functions assuming
    # that they are C++ functions (i.e. the functions are not declared as
    # ``extern "C" {...}``).
    #
    #   - The docs says that it's okay if the C functions have C++ entry points
    #   - but otherwise, they recommend writing a small C++ shim module
    #
    # It's not exactly clear what "C++ entry-points" mean... But, I think this
    # is okay?

    bint calc_vsf_props(const PointProps points_a, const PointProps points_b,
                        const char * pairwise_op, void* accumhandle,
                        const double *bin_edges, size_t nbins,
                        const ParallelSpec parallel_spec)

    bint cxx_compiled_with_openmp "compiled_with_openmp"()

cdef extern from "accum_handle.hpp":

    void* accumhandle_create(const StatListItem* stat_list,
                             size_t stat_list_len,
                             size_t num_dist_bins)

    void accumhandle_destroy(void* handle)

    void accumhandle_export_data(void* handle, double *out_flt_vals,
                                 int64_t *out_i64_vals)

    void accumhandle_restore(void* handle, const double *in_flt_vals,
                             const int64_t *in_i64_vals)

    void accumhandle_consolidate_into_primary(void* handle_primary,
                                              void* handle_secondary)

    void accumhandle_add_entries(void* handle, int purge_everything_first,
                                 size_t spatial_bin_index, size_t num_entries,
                                 double * values, double * weights)

    void accumhandle_postprocess(void* handle)

    const char* accumhandle_prop_name(void* handle, int i)

    int accumhandle_prop_isdouble(void* handle, int i)

    int accumhandle_prop_count(void* handle, int i)


def compiled_with_openmp():
    return bool(cxx_compiled_with_openmp())

cdef double* _array_to_ptr(object arr) except*:

    cdef double[:, ::1] view_2D
    cdef double[::1] view_1D
    cdef double* out
    if arr is None:
        out = NULL
    elif arr.ndim == 1:
        view_1D = arr
        out = &view_1D[0]
    elif arr.ndim == 2:
        view_2D = arr
        out = &view_2D[0,0]
    else:
        raise ValueError("arr must be None, 1D or 2D")
    return out

cdef class PyPointsProps:
    """A simple wrapper class

    The structe draws a lot of inspiration from
    https://docs.cython.org/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers
    """

    cdef PointProps c_points # wrapped c++ instance
    # the following attribute is intended to hold the numpy arrays that manage
    # of the arrays are consistent with the rest of the object
    cdef list refs

    def __init__(self):
        # Prevent instantiation from normal Python code
        raise TypeError("This class cannot be instantiated directly.")

    @staticmethod
    cdef PyPointsProps factory(PointProps c_points, list refs):
        """factory method"""
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef PyPointsProps out = PyPointsProps.__new__(PyPointsProps)
        out.c_points = c_points
        out.refs = refs
        return out

    @property
    def n_points(self): return self.c_points.n_points

    @property
    def n_spatial_dims(self):  return self.c_points.n_spatial_dims

    @property
    def spatial_dim_stride(self):  return self.c_points.spatial_dim_stride

    def has_weights(self):
        return self.c_points.weights != NULL

    def has_non_positive_weights(self):
        if self.c_points.weights == NULL:
            return False
        cdef Py_ssize_t i
        for i in range(self.c_points.n_points):
            if self.c_points.weights[i] <= 0:
                return True
        return False

def _construct_pointprops(pos, val, weights = None, val_is_vector = True,
                          dtype = np.float64, allow_null_contents = False):
    """A Helper function that is used to construct the point properties"""
    cdef PointProps c_points
    assert np.dtype(dtype) == np.float64

    if allow_null_contents:
        # just used to format errors
        pos_name, val_name, weights_name = "pos_a", "val_a", "weights_a"
    else:
        pos_name, val_name, weights_name = "pos_b", "val_b", "weights_b"

    # handle the 2 simple cases
    if allow_null_contents and (pos is None) and (val is None):
        if weights is not None:
            raise ValueError(f"{weights_name} must be None when {pos_name} is None")

        # initialize the c_points struct
        c_points.positions = NULL
        c_points.values = NULL
        c_points.weights = NULL
        c_points.n_points = 0
        c_points.n_spatial_dims = 0
        c_points.spatial_dim_stride = 0
        return PyPointsProps.factory(c_points, [])

    elif (pos is None) or (val is None):
        raise ValueError(f"{pos_name} and {val_name} must not be None")

    # down here, we handle the general case:

    pos_arr = np.asarray(pos, dtype = dtype, order = 'C')
    if pos_arr.ndim != 2:
        raise ValueError(f"the only valid array shape for {pos_name} is 2D")

    n_spatial_dims, n_points = pos_arr.shape

    val_arr = np.asarray(val, dtype = dtype, order = 'C')
    if val_is_vector and (pos_arr.shape != val_arr.shape):
        if val_arr.ndim != 2:
            raise ValueError(f"since {val_name} specifies a vector, it must be 2D")
        raise ValueError(
            f"since {pos_name} specifies {n_points} points and {val_name} represents "
            f"a vector, {val_name} must be an array of shape {pos_arr.shape}"
        )
    elif (not val_is_vector) and ((n_points,) != val_arr.shape):
        if val_arr.ndim != 1:
            raise ValueError(f"since {val_name} specifies a scalar, it must be 1D")
        raise ValueError(
            f"since {pos_name} specifies {n_points} points and {val_name} represents "
            f"a scalar, {val_name} must be an array of shape ({n_points},)"
        )

    # Here we perform some sanity checks on the length scale
    # -> these first 2 checks may be redundant with np.asarray(..., order = 'C')
    assert pos_arr.strides[1] == pos_arr.itemsize
    assert val_arr.strides[-1] == val_arr.itemsize

    # in the future, consider relaxing the following conditions (to maybe
    # facilitate better data alignment)
    assert pos_arr.strides[0] == (n_points * pos_arr.itemsize)
    if val_is_vector:
        assert val_arr.strides[0] == (n_points * val_arr.itemsize)
    spatial_dim_stride = int(n_points)


    weights_arr = None
    if weights is not None:
        weights_arr = np.asarray(weights, dtype = dtype, order = 'C')
        if weights_arr.shape != (n_points,):
            raise ValueError(
                f"since {pos_name} specifies {n_points} points, {weights_name} "
                f"must be an array of shape ({n_points},) or it must be None"
            )

    # initialize most of the c_points struct
    c_points.positions = _array_to_ptr(pos_arr)
    c_points.values = _array_to_ptr(val_arr)
    c_points.weights = _array_to_ptr(weights_arr)
    c_points.n_points = n_points
    c_points.n_spatial_dims = n_spatial_dims
    c_points.spatial_dim_stride = spatial_dim_stride
    return PyPointsProps.factory(c_points, [pos_arr, val_arr, weights_arr])


cdef object _process_statistic_args(object statconf_l, object dist_bin_edges,
                                    void** accumhandle):
    """
    Construct the accumhandle as well as information about the output data

    Since we return a python object, cython always checks for exceptions when this
    returns. (cython issues a warning if you add an ``except*`` clause)
    """

    # we sort the entries of statconf_l in alphabetical order because all of the fused
    # accumulators expect the statnames to be in that order...
    # -> (in the future, we will hopefully remove fused accumulators)
    # -> index_statconf_pairs holds the pairs (original_index, statconf)
    index_statconf_pairs = sorted(enumerate(statconf_l),
                                  key = lambda pair: pair[1].name)
    sorted_statconf_l = list(zip(*index_statconf_pairs))[1]

    # create the accumulator
    accumhandle[0] = _construct_accum_handle(dist_bin_edges, sorted_statconf_l)

    key_groups = [None for i in range(len(sorted_statconf_l))]
    dtype_props = {np.int64 : [], np.float64 : []}
    for original_index, statconf in index_statconf_pairs:
        cur_key_group = []
        for quan_name, dtype, shape in statconf.get_dset_props(dist_bin_edges):
            cur_key_group.append( f'{statconf.name}-{quan_name}' )
            dtype_props[dtype].append((cur_key_group[-1], dtype, shape))
        key_groups[original_index] = cur_key_group

    all_props = dtype_props[np.int64] + dtype_props[np.float64]
    return ArrayMap(all_props), key_groups

def _coerce_stat_list(arg):
    argname = 'stat_kw_pairs'


    def is_statconf(elem):
        return isinstance(elem, StatConf)

    def is_statkw_pair(elem):
        try:
            length = len(elem)
        except TypeError:
            return False
        return (
            (length == 2) and isinstance(elem[0], str) and isinstance(elem[1], dict)
        )

    err_prefix = (
        f"the elements of {argname} must be StatConf instances or 2-tuples specifying "
        "statname-kwarg pairs (i.e. a str-dict pair):"
    )

    try:
        first_elem = arg[0]
    except TypeError:
        raise TypeError(f"{argname} must be a sequence") from None
    except IndexError:
        raise ValueError(f"{argname} must be a non-empty sequence") from None

    if is_statconf(first_elem):
        for i, elem in enumerate(arg[1:]):
            if is_statkw_pair(elem):
                raise ValueError(f"{err_prefix} You can't mix!")
            elif not is_statconf(elem):
                raise TypeError(f"{err_prefix} element {i+1}, {elem!r}, has wrong type")
        return arg

    elif is_statkw_pair(first_elem):
        out = [get_statconf(*first_elem)]
        for i,elem in enumerate(arg[1:]):
            if is_statconf(elem):
                raise ValueError(f"{err_prefix} You can't mix!")
            elif not is_statkw_pair(elem):
                raise TypeError(f"{err_prefix} element {i+1}, {elem!r}, has wrong type")
            out.append(get_statconf(*elem))
        return out

    else:
        raise TypeError(f"{err_prefix} Element 0, {first_elem!r} has the wrong type")

def _validate_stat_kw_pairs(arg):
    if not isinstance(arg, Sequence):
        raise ValueError("stat_kw_pairs must be a sequence")

    for elem in arg:
        if len(elem) != 2:
            raise ValueError("Each element in stat_kw_pairs must hold 2"
                             "elements")
        first, second = elem
        if (not isinstance(first, str)) or (not isinstance(second, dict)):
            raise ValueError("Each element in stat_kw_pairs must hold a "
                             "string paired with a dict")


_KNOWN_OP = frozenset([
    "vector_diff",
    "longitudinal_diff",
    "scalar_correlate",
    "vector_correlate",
    "longitudinal_correlate"
])


def _core_pairwise_work(pos_a, pos_b, val_a, val_b, dist_bin_edges,
                        weights_a = None, weights_b = None,
                        pairwise_op = "vector_diff",
                        stat_kw_pairs = [('variance', {})],
                        nproc = 1, force_sequential = False,
                        postprocess_stat = True):
    statconf_l = _coerce_stat_list(stat_kw_pairs)

    if pairwise_op not in _KNOWN_OP:
        raise ValueError(
            f"{pairwise_op!r} is not a known pairwise_op. Known values "
            f"include be one of {tuple(_KNOWN_OP)}"
        )

    val_is_vector = "scalar" not in pairwise_op
    cdef PyPointsProps points_a = _construct_pointprops(
            pos_a, val_a, weights = weights_a, val_is_vector = val_is_vector,
            dtype = 'f8', allow_null_contents = False)
    cdef PyPointsProps points_b = _construct_pointprops(
            pos_b, val_b, weights = weights_b, val_is_vector = val_is_vector,
            dtype = 'f8', allow_null_contents = True)

    # do some basic argument checking
    if (pos_b is None) and (points_a.n_points <= 1):
        raise ValueError("When pos_b and vel_b are None, then pos_a and vel_a "
                         "must specify properties for more than 1 point")
    elif ((pos_b is not None) and
          (points_a.n_spatial_dims != points_b.n_spatial_dims)):
        raise ValueError("When pos_a and pos_b are both specified, they must "
                         "have consistent spatial dimensions")
    elif points_a.n_spatial_dims != 3:
        raise NotImplementedError(
            "vsf_props currently only has support for computing velocity "
            "structure function properties for sets of points with 3 spatial "
            "dimensions"
        )
    elif (pos_b is not None) and ((weights_a is None) != (weights_b is None)):
        raise ValueError(
            "when pos_b is not None, then you must either: \n"
            "  - set weights_a and weights_b to None OR\n"
            "  - provide non-None values for both weights_a and weights_b")
    elif (points_a.has_non_positive_weights() or
          points_b.has_non_positive_weights()):
        raise ValueError("you can't provide non-positive weights")

    # check if any statistics require weights
    requires_weights = any(statconf.requires_weights for statconf in statconf_l)
    if requires_weights and (weights_a is None):
        raise ValueError("one of the statistics requires weights, but no "
                         "weights were provided")
    elif (not requires_weights) and (weights_a is not None):
        raise ValueError("it is an error to provide weights when no stats "
                         "require them")

    # check validity of dist_bin_edges (and do any necessary coercion)
    dist_bin_edges = np.asanyarray(dist_bin_edges, dtype = np.float64)
    if not dist_bin_edges.flags['C_CONTIGUOUS']:
        dist_bin_edges = np.ascontiguousarray(dist_bin_edges)
    if not _verify_bin_edges(dist_bin_edges):
        raise ValueError(
            'dist_bin_edges must be a 1D monotonically increasing array with '
            '2 or more values'
        )
    ndist_bins = dist_bin_edges.size - 1
    cdef const double[::1] bin_edges_view = dist_bin_edges

    # construct stat_list and rslt_container
    cdef void* accumhandle = NULL
    rslt_container, key_groups = _process_statistic_args(statconf_l, dist_bin_edges,
                                                         &accumhandle)

    cdef ParallelSpec parallel_spec
    parallel_spec.nproc = nproc
    parallel_spec.force_sequential = force_sequential

    cdef bytes casted_pairwise_op = pairwise_op.encode("ASCII")
    cdef const char* c_pairwise_op = casted_pairwise_op

    cdef bint success = calc_vsf_props(
        points_a = points_a.c_points, points_b = points_b.c_points,
        pairwise_op = c_pairwise_op,
        accumhandle = accumhandle,
        bin_edges = &(bin_edges_view[0]), nbins = ndist_bins,
        parallel_spec = parallel_spec
    )

    if not success:
        raise RuntimeError("Something went wrong while in calc_vsf_props")

    _export_to_ArrayMap_from_handle(accumhandle, rslt_container)

    out = []
    for statconf, key_grp in zip(statconf_l, key_groups):
        name_len = len(statconf.name)
        val_dict = {k[name_len+1:] : rslt_container[k] for k in key_grp}

        if postprocess_stat:
            statconf.postprocess_rslt(val_dict)
        out.append(val_dict)

    accumhandle_destroy(accumhandle)
    return out

# this is an object used to denote that an argument wasn't provided while we
# deprecate an old interface
_unspecified = object()

# as in twopoint_correlation, you can use val_a, val_b, dist_bin_edges as
# positional arguments
def vsf_props(pos_a, pos_b, *args, val_a = _unspecified, val_b = _unspecified,
              vel_a = _unspecified, vel_b = _unspecified,
              dist_bin_edges = _unspecified,
              weights_a = None, weights_b = None,
              stat_kw_pairs = [('variance', {})],
              longitudinal = False,
              nproc = 1, force_sequential = False,
              postprocess_stat = True):
    """
    Calculates properties pertaining to the vector structure function for
    pairs of points. It's commonly used for the velocity structure function in
    particular.

    If you set both ``pos_b`` and ``val_b`` to ``None`` then the structure
    function properties will only be computed for unique pairs of the
    points specified by ``pos_a`` and ``val_a``

    Parameters
    ----------
    pos_a, pos_b : array_like
        2D arrays holding the positions of each point. Axis 0 should be the
        number of spatial dimensions must be consistent for each array. Axis 1
        can be different for each array
    val_a, val_b : array_like
        2D arrays holding the vector values at each point. The shape of
        ``val_a`` should match ``pos_a`` and the shape of ``val_b`` should
        match ``pos_b``.
    dist_bin_edges : array_like
        1D array of monotonically increasing values that represent edges for
        distance bins. A distance ``x`` lies in bin ``i`` if it lies in the
        interval ``dist_bin_edges[i] <= x < dist_bin_edges[i+1]``.
    weights_a, weights_b : array_like, optional
        optional 1D arrays that can be used to specify weights for point. When
        specified, the size of ``weights_a`` should match
        ``np.shape(pos_a)[1]`` and the size of ``weights_b`` should match
        ``np.shape(pos_b)[1]``. It is an error to specify weights when the
        specified statistics won't use them.
    stat_kw_pairs : sequence of (str, dict) tuples
        Each entry is a tuple holding the name of a statistic to compute and a
        dictionary of kwargs needed to compute that statistic. A list of valid
        statistics are described below. Unless we explicitly state otherwise,
        an empty dict should be passed for the kwargs.
    nproc : int, optional
        Number of processes to use for parallelizing this calculation. Default
        is 1. If the problem is small enough, the program may ignore this
        argument and use fewer processes.
    force_sequential : bool, optional
        `False` by default. When `True`, this forces the code to run with a
        single process (regardless of the value of `nproc`). However, the data
        is still partitioned as though it were using `nproc` processes. Thus,
        floating point results should be bitwise identical to an identical
        function call where this is `False`. (This is primarily provided for
        debugging purposes)
    postprocess_stat : bool, optional
        Users directly employing this function should almost always set this
        kwarg to `True` (the default). This option is only provided to simplify
        the process of consolidating results from multiple calls to vsf_props.
    vel_a, vel_b : array_like
        Parameters that are deprecated in favor of ``val_a`` and ``val_b``.

    Notes
    -----
    Currently recognized unweighted statistic names include:
        - ``'mean'`` : calculate the 1st order structure function.
        - ``'variance'`` : calculate the 1st order structure function and
          the variance (while variance is related to the 2nd order
          structure function, it is NOT the same)
        - ``'omoment2'`` : calculate the 1st and 2nd order structure
          functions.
        - ``'omoment3'`` : calculate the 1st, 2nd, and 3rd order structure
          functions
        - ``'omoment4'`` : calculate the 1st, 2nd, 3rd, and 4th order
          structure functions
        - ``'histogram'`` : this constructs a 2D histogram. The bin edges
          along axis 0 are given by the `dist_bin_edges` argument. The
          magnitudes of the vector differences are binned along axis 1.
          The 'val_bin_edges' keyword must be specified alongside this
          statistic name (to specify the bin edges along axis 1). It
          should be associated with a 1D monotonic array.

    Weighted versions of each of these statistics are also available. To
    access these, you should prepend ``"weighted"`` to the start of the
    string (so ``"weightedmean"`` instead of ``"mean"`` or
    ``"weightedhistogram"`` instead of ``"histogram"``).

    **BE AWARE**, that unlike ``'variance'``, ``'weightedvariance'`` does
    **NOT** attempt to make any corrections to get an unbiased estimate of
    variance.
    """


    # do some messy work to help us deprecate vel_a and vel_b

    # Step 1: we do some basic preparation
    is_provided = lambda arg: arg is not _unspecified
    _names = ("val_a", "val_b", "dist_bin_edges")
    _val_a, _val_b = _unspecified, _unspecified
    if is_provided(val_a) or is_provided(val_b):
        if is_provided(vel_a) or is_provided(vel_b):
            raise ValueError("Don't mix val_a,val_b with vel_a,vel_b")
        _val_a, _val_b = val_a, val_b
    elif is_provided(vel_a) or is_provided(vel_b):
        _val_a, _val_b = vel_a, vel_b
        _names = ("vel_a", "vel_b", "dist_bin_edges")

    # Step 2: do the main checks
    if is_provided(_val_a):
        if len(args) != 0:
            raise ValueError(f"the {_names[0]} argument was specified more "
                             "than once")
        elif _val_b is _unspecified:
            raise ValueError(f"missing the {_names[1]} argument")
        elif dist_bin_edges is _unspecified:
            raise ValueError(f"missing the {_names[2]} argument")
        # do nothing

    elif is_provided(_val_b):  # _val_a is NOT a kwarg
        if len(args) > 1:
            raise ValueError(f"the {_names[1]} argument was specified more "
                             "than once")
        elif len(args) == 0:
            raise ValueError(f"missing the {_names[0]} argument")
        elif dist_bin_edges is _unspecified:
            raise ValueError(f"missing the {_names[2]} argument")
        _val_a = args[0]

    elif is_provided(dist_bin_edges):  # _val_a & _val_b are NOT kwargs
        if len(args) > 2:
            raise ValueError(f"the {_names[2]} argument was specified more "
                             "than once")
        elif len(args) < 2:
            raise ValueError(f"missing the {_names[len(args)]} argument")
        _val_a, _val_b = args

    else:  # _val_a, _val_b, & dist_bin_edges are NOT kwargs
        if len(args) > 3:
            raise ValueError("received too many positional arguments")
        _val_a, _val_b, dist_bin_edges = args

    # Step 3: Warn people if they use deprecated kwargs
    if "vel_a" in _names:
        warnings.warn(
            "The vel_a and vel_b kwargs are deprecated in favor of val_a and "
            "val_b", DeprecationWarning)

    # sanity check
    assert _val_a is not _unspecified
    assert _val_b is not _unspecified
    assert dist_bin_edges is not _unspecified

    if longitudinal:
        pairwise_op = "longitudinal_diff"
    else:
        pairwise_op = "vector_diff"

    return _core_pairwise_work(
        pos_a = pos_a, pos_b = pos_b, val_a = _val_a, val_b = _val_b,
        dist_bin_edges = dist_bin_edges,
        weights_a = weights_a, weights_b = weights_b,
        pairwise_op = pairwise_op,
        stat_kw_pairs = stat_kw_pairs, nproc = nproc,
        force_sequential = force_sequential,
        postprocess_stat = postprocess_stat)


def twopoint_correlation(pos_a, pos_b, val_a, val_b, dist_bin_edges,
                         *, stat_kw_pairs = [('mean', {})],
                         longitudinal = False,
                         nproc = 1, force_sequential = False):
    """
    Calculates the 2pcf (two-point correlation function) for pairs of points.

    If you set both ``pos_b`` and ``val_b`` to ``None`` then the two-point
    correlation function will only be computed for unique pairs of the points
    specified by ``pos_a`` and ``val_a``

    Parameters
    ----------
    pos_a, pos_b : array_like
        2D arrays holding the positions of each point. Axis 0 should be the
        number of spatial dimensions must be consistent for each array. Axis 1
        can be different for each array
    val_a, val_b : array_like
        These can be 1D arrays holding the velocities at each point. In that case, the
        size of ``val_a`` should match the length of ``pos_a`` along axis 0 and the
        and size of ``val_b`` should match the the length of ``pos_b``. Alternatively,
        these can be 2D arrays. In this case, the shape of ``val_a`` should match
        ``pos_a`` and the shape of ``val_b`` should match ``pos_b``.
    dist_bin_edges : array_like
        1D array of monotonically increasing values that represent edges for
        distance bins. A distance ``x`` lies in bin ``i`` if it lies in the
        interval ``dist_bin_edges[i] <= x < dist_bin_edges[i+1]``.
    stat_kw_pairs : sequence of (str, dict) tuples, optional
        The default choice is most meaningful for the 2pcf. In practice, this
        can accept the same arguments (other than the weighted arguments)
        accepted by :py:func:`vsf_props`.
    nproc : int, optional
        Number of processes to use for parallelizing this calculation. Default
        is 1. If the problem is small enough, the program may ignore this
        argument and use fewer processes.
    force_sequential : bool, optional
        `False` by default. When `True`, this forces the code to run with a
        single process (regardless of the value of `nproc`). However, the data
        is still partitioned as though it were using `nproc` processes. Thus,
        floating point results should be bitwise identical to an identical
        function call where this is `False`. (This is primarily provided for
        debugging purposes)

    Notes
    -----
    Currently recognized statistic names include:
        - ``'mean'``: the typical correlation function
        - ``'variance'``: the variance of the products of the pairs of
          scalar values are computed for all pairs of values in a given
          distance bin (in addition to ``'mean'``).
        - ``'omoment2'``: calculates the 2nd order moment about the origin
          for all pairs of points in a a given distance bin (in addition
          to ``'mean'``).
        - ``'omoment3'``: calculates the 3rd order moment about the origin
          for all pairs of points in a a given distance bin (in addition
          to ``'mean'`` and ``'omoment2'``).
        - ``'omoment4'``: calculates the 4th order moment about the origin
          for all pairs of points in a a given distance bin (in addition
          to ``'mean'``, ``'omoment2'``, and ``'omoment3'``).
        - 'histogram': this constructs a 2D histogram. The bin edges along
          axis 0 are given by the `dist_bin_edges` argument. The products
          of the pairs of scalar values are binned along axis 1. The
          'val_bin_edges' keyword must be specified alongside this
          statistic name (to specify the bin edges along axis 1). It
          should be associated with a 1D monotonic array.


    Weighted versions of each of these statistics are also available. To
    access these, you should prepend ``"weighted"`` to the start of the
    string (so ``"weightedmean"`` instead of ``"mean"`` or
    ``"weightedhistogram"`` instead of ``"histogram"``).

    **BE AWARE**, that unlike ``'variance'``, ``'weightedvariance'`` does
    **NOT** attempt to make any corrections to get an unbiased estimate of
    variance.
    """
    if longitudinal:
        pairwise_op = "longitudinal_correlate"
    elif np.ndim(val_a) == 2:
        pairwise_op = "vector_correlate"
    else:
        pairwise_op = "scalar_correlate"

    return _core_pairwise_work(
        pos_a = pos_a, pos_b = pos_b, val_a = val_a, val_b = val_b,
        dist_bin_edges = dist_bin_edges, weights_a = None, weights_b = None,
        pairwise_op = pairwise_op, stat_kw_pairs = stat_kw_pairs, nproc = nproc,
        force_sequential = force_sequential, postprocess_stat = True)


cdef BinSpecification _build_BinSpecification(arr, wrap_array = True):
    if not _verify_bin_edges:
        raise ValueError('arr must be a 1D monotonically increasing array with '
                         '2 or more values')

    cdef BinSpecification out
    out.n_bins = <size_t>(arr.size - 1)

    cdef double[::1] arr_memview

    if wrap_array:
        assert arr.dtype == np.float64
        assert arr.flags['C_CONTIGUOUS']
        arr_memview = arr

        out.bin_edges = &arr_memview[0]
    else:
        raise RuntimeError("Not implemented yet!")
    return out


cdef bytes _coerce_name_to_bytes(name) except*:
    if isinstance(name, str):
        return name.encode('ASCII')
    elif isinstance(name, (bytes, bytearray)):
        return bytes(name)
    else:
        raise TypeError("name must have type: str, bytes, or bytearray")


cdef void* _construct_accum_handle(object dist_bin_edges, object statconf_l) except NULL:
    assert PY_MAJOR_VERSION >= 3

    if not isinstance(statconf_l, (list,tuple)):
        statconf_l = [statconf_l]

    assert len(statconf_l) in [1,2]

    cdef size_t num_dist_bins = dist_bin_edges.size - 1

    # define variables used to hold statistic names
    cdef bytes coerced_name_str0
    cdef bytes coerced_name_str1

    # allocate space for variables that may be used for bin specification
    cdef BinSpecification[2] bin_spec

    # allocate space for variables that may be used for stat list
    cdef StatListItem[2] entries

    cdef Py_ssize_t i
    for i,statconf in enumerate(statconf_l):
        name = statconf.name
        kwargs = statconf._kwargs()
        if 'val_bin_edges' in kwargs:
            assert len(kwargs) == 1
            val_bin_edges = kwargs['val_bin_edges']
        else:
            assert len(kwargs) == 0
            val_bin_edges = None

        if i == 0:
            # lifetime of str is tied to the coerced_name_str0 variable
            coerced_name_str0 = _coerce_name_to_bytes(name)
            entries[0].statistic = coerced_name_str0
        else:
            # lifetime of str is tied to the coerced_name_str1 variable
            coerced_name_str1 = _coerce_name_to_bytes(name)
            entries[1].statistic = coerced_name_str1

        if val_bin_edges is not None:
            # lifetime of bin_spec is tied to statconf
            bin_spec[i] = _build_BinSpecification(val_bin_edges, True)
            entries[i].arg_ptr = <void*>(bin_spec+i)
        else:
            entries[i].arg_ptr = NULL

    return accumhandle_create(entries, len(statconf_l), num_dist_bins)


cdef int64_t* _ArrayMap_i64_ptr(object array_map):
    cdef object i64_array = array_map.get_int64_buffer()
    if i64_array.size == 0:
        return NULL
    cdef int64_t[::1] i64_vals = i64_array
    return &i64_vals[0]

cdef double* _ArrayMap_flt_ptr(object array_map):
    cdef object flt_array = array_map.get_float64_buffer()
    if flt_array.size == 0:
        return NULL
    cdef double[::1] flt_vals = flt_array
    return &flt_vals[0]

cdef void _restore_handle_from_ArrayMap(void* handle, object array_map):
    accumhandle_restore(handle,
                        _ArrayMap_flt_ptr(array_map),
                        _ArrayMap_i64_ptr(array_map))

cdef void _export_to_ArrayMap_from_handle(void* handle, object array_map):
    accumhandle_export_data(handle,
                            _ArrayMap_flt_ptr(array_map),
                            _ArrayMap_i64_ptr(array_map))

cdef class SFConsolidator:
    """
    This performs accumulation using the accumhandle objects
    """

    cdef void* primary_handle
    cdef void* secondary_handle
    cdef object statconf
    cdef object kwargs
    cdef object dist_bin_edges

    def __cinit__(self, object dist_bin_edges, object statconf):
        cdef object kwargs = statconf._kwargs()
        self.primary_handle = _construct_accum_handle(dist_bin_edges, statconf)
        self.secondary_handle = _construct_accum_handle(dist_bin_edges, statconf)
        self.statconf = statconf
        self.kwargs = kwargs
        self.dist_bin_edges = dist_bin_edges

    def __dealloc__(self):
        accumhandle_destroy(self.primary_handle)
        accumhandle_destroy(self.secondary_handle)

    def _get_entry_spec(self):
        return self.statconf.get_dset_props(self.dist_bin_edges)

    def _purge_values(self):
        # the choice of spatial_bin_index doesn't matter since we don't actually add
        # any values. We're just taking advantage of the ability to zero-initialize
        # the values
        spatial_bin_index = 0
        accumhandle_add_entries(self.primary_handle, 1, spatial_bin_index,
                                0, NULL, NULL)

    def consolidate(self, *rslts):
        # first lets purge the values held in primary_handle
        self._purge_values()

        cdef object tmp = ArrayMap(self._get_entry_spec())
        for rslt in rslts:
            if len(rslt) == 0:
                continue

            # load data from rslt into self.secondary_handle
            if isinstance(rslt, ArrayMap):
                _restore_handle_from_ArrayMap(self.secondary_handle, rslt)
            else:
                for key in tmp:
                    tmp[key][...] = rslt[key]
                _restore_handle_from_ArrayMap(self.secondary_handle, tmp)

            # update self.primary_handle
            accumhandle_consolidate_into_primary(self.primary_handle,
                                                 self.secondary_handle)
        # export data from self.primary_handle
        _export_to_ArrayMap_from_handle(self.primary_handle, tmp)
        return tmp.asdict()

def consolidate_partial_results(statconf, results, dist_bin_edges):
    """
    This function is used to consolidate the partial results from multiple
    executions of the `vsf_props` or `twopoint_correlation function.
    """
    if len(results) == 0:
        raise ValueError("Can't consolidate 0 results")
    consolidator = SFConsolidator(dist_bin_edges=dist_bin_edges, statconf=statconf)
    return consolidator.consolidate(*results)

def _test_evaluate_statconf(statconf, values, weights = None):
    """
    This exists for testing purposes (to let us check whether"""
    dist_bin_edges = np.array([0.0, 1.0])
    if statconf.requires_weights and weights is None:
        raise ValueError("The specified statconf requires that weights is provided")
    elif np.ndim(values) != 1.0:
        raise ValueError("values must be a 1d array")
    elif np.size(values) == 0:
        raise ValueError("values must hold at least 1 element")
    elif (weights is not None) and np.shape(weights) != np.shape(values):
        raise ValueError("when specified, weights must have the same shape as values")

    cdef double[::1] values_view
    cdef double[::1] weights_view

    cdef object out = ArrayMap(statconf.get_dset_props(dist_bin_edges))
    cdef void* handle = _construct_accum_handle(dist_bin_edges, statconf)

    try:
        values = np.asarray(values, dtype = np.float64, order = 'C')
        values_view = values

        if statconf.requires_weights:
            weights = np.asarray(weights, dtype = np.float64, order = 'C')
            weights_view = weights
            accumhandle_add_entries(handle, 0, 0, values.size, &values_view[0],
                                    &weights_view[0])
        else:
            accumhandle_add_entries(handle, 0, 0, values.size, &values_view[0], NULL)
        _export_to_ArrayMap_from_handle(handle, out)
    finally:
        accumhandle_destroy(handle)
    return out

def _validate_basic_quan_props(statconf, rslt, dist_bin_edges):
    """Helper function that performs some basic checks"""
    quan_props = statconf.get_dset_props(dist_bin_edges)
    if len(quan_props) != len(rslt):
        raise ValueError("The rslt doesn't have the expected number of entries")
    for name, dtype, shape in quan_props:
        if name not in quan_props:
            raise ValueError(
                f"The result for the '{statconf.name}' statistic is missing a "
                f"quantity called '{name}'"
            )
        elif rslt[name].dtype != dtype:
            raise ValueError(
                f"the {name} quantity for the {statconf.name} statistic should ",
                f"have a dtype of {dtype}, not of {rslt[name].dtype}"
            )
        elif rslt[name].shape != shape:
            raise ValueError(
                f"the {name} quantity for the {statconf.name} statistic should ",
                f"have a shape of {shape}, not of {rslt[name].shape}"
            )

def _validate_counts_or_weights(statconf, rslt, key, *, max_total_count = None):
    """Helper function that checks validity of the counts or weights key"""
    weights_arr = rslt[key]
    # first check for negative values
    if np.any(weights_arr < 0):
        raise ValueError(
            f"the '{key}' result for the '{statconf.name}' statistic can't contain "
            "negative values"
        )

    # in the event that we are using counts and the caller provides the max pairs
    # (an upper limit on the number of pairs that could influence the result), we
    # can check that the number of counts doesn't exceed max_pairs
    if (not statconf.requires_weights) and (max_total_count is not None):
        counts = weights_arr
        if max_total_count > np.iinfo(np.int64).max:
            count_tot = sum(int(e) for e in counts.flat)
        else:
            count_tot = np.sum(counts, dtype = np.int64)
        if count_tot > max_total_count:
            raise ValueError(
                f"The total value of the '{key}' result for the '{statconf.name}' "
                f"statistic indicates that a total of {count_tot} pairs of points "
                "were used to compute the result. The max expected pairs of points "
                f"is only {max_total_count}."
            )

def _infer_dist_bin_count(statconf, rslt_dict):
    # hacky function to infer the number of distance bins used by the accumulator that
    # constructed rslt_dict
    dummy_dist_bin_edges = np.array([0.0, 1.0])
    first_dset_prop = statconf.get_dset_props(dummy_dist_bin_edges)[0]
    elements_per_dist_bin = np.prod(first_dset_prop.shape)
    nelements = rslt_dict[first_dset_prop.name].size
    n_dist_bins = nelements // elements_per_dist_bin
    assert (elements_per_dist_bin * n_dist_bins) == nelements  # sanity check!
    return n_dist_bins


def _postprocess(statconf, rslt_dict):
    # first, we need to infer the number of distance bins
    dummy_dist_bin_edges = np.arange(
        _infer_dist_bin_count(statconf, rslt_dict) + 1, dtype=np.float64
    )

    rslt_is_ArrayMap = isinstance(rslt_dict, ArrayMap)
    if rslt_is_ArrayMap:
        arr_map = rslt_dict
    else:
        arr_map = ArrayMap(statconf.get_dset_props(dummy_dist_bin_edges))
        for key in arr_map:
            arr_map[key][...] = rslt_dict[key]

    cdef void* handle = _construct_accum_handle(dummy_dist_bin_edges, [statconf])
    _restore_handle_from_ArrayMap(handle, arr_map)
    accumhandle_postprocess(handle)
    _export_to_ArrayMap_from_handle(handle, arr_map)
    accumhandle_destroy(handle)

    if not rslt_is_ArrayMap:
        for key in arr_map:
            rslt_dict[key][...] = arr_map[key]

class ExperimentalWarning(UserWarning):
    pass

def _get_prop_descr_l(statconf):

    out = []
    dist_bin_edges = np.array([0.0, 1.0])
    cdef const char* name
    cdef int is_f64, count
    cdef int i = 0
    cdef void* handle = _construct_accum_handle(dist_bin_edges, statconf)
    try:
        while True:
            name = accumhandle_prop_name(handle, i)
            if name == NULL:
                break
            is_f64 = accumhandle_prop_isdouble(handle, i)
            count = accumhandle_prop_count(handle, i)
            if is_f64:
                out.append(((<bytes>name).decode('ascii'), np.float64, int(count)))
            else:
                out.append(((<bytes>name).decode('ascii'), np.int64, int(count)))
            i+=1
    finally:
        accumhandle_destroy(handle)
    return out

class _DSetProp(NamedTuple):
    name : str
    dtype : Type[Union[np.float64, np.int64]]
    shape : Tuple[int, int]

_LEGACY_PROPNAME_REMAP = {
    "mean" : ('counts', "mean"),
    "weightedmean" : (None, "mean"),
    "variance" : ('counts', "mean", "variance"),
    "weightedvariance" : (None, "mean", "variance"),
    "histogram" : ('2D_counts',),
    "weightedhistogram" : ("2D_weight_sums",)
}

_OTHER_NAMES = [f"omoment{i}" for i in [2,3,4]]

_ALL_SF_STAT_NAMES = (
    tuple(name for name in _LEGACY_PROPNAME_REMAP) +
    tuple(_OTHER_NAMES) +
    tuple(f'weighted{name}' for name in _OTHER_NAMES)
)

def _get_dset_props_helper(statconf, dist_bin_edges, use_legacy_names = True):
    """
    Helper function that returns the dset_props and the index of
    dset_props that corresponds to "weight_sum"
    """
    _check_dist_bin_edges(dist_bin_edges)

    remap_list = None
    if use_legacy_names:
        remap_list = _LEGACY_PROPNAME_REMAP.get(statconf.name, None)

    num_dist_bins = np.size(dist_bin_edges) - 1
    def get_shape(count):
        if count == 1:
            return (num_dist_bins,)
        return (num_dist_bins, count)

    tmp = _get_prop_descr_l(statconf)

    if remap_list is None:
        dset_props = [_DSetProp(name, dtype, get_shape(count))
                      for (name, dtype, count) in tmp]
    else:
        assert len(remap_list) == len(tmp)
        dset_props = []
        for i, (name, dtype, count) in enumerate(tmp):
            if remap_list[i] is not None:
                name = remap_list[i]
            dset_props.append(_DSetProp(name, dtype, get_shape(count)))
    count_weight_index = 0
    return dset_props


class StatConf:
    """
    This class is used to represent configurations of structure-function/correlation
    function statistics.
    """

    def __init__(self, name, kwargs):
        if name not in _ALL_SF_STAT_NAMES:
            raise ValueError(f"There is no statistic known as `{name}`")
        elif "histogram" in name:
            if (kwargs is None) or (list(kwargs.keys()) != ['val_bin_edges']):
                raise ValueError("'val_bin_edges' is required as the single kwarg for "
                                 "computing histogram-statistics")
            _check_bin_edges_arg(kwargs['val_bin_edges'], "'val_bin_edges' kwarg")
        elif (kwargs is not None) and (len(kwargs) > 0):
            raise ValueError(f"the {name} stat should have no kwargs")
        else:
            kwargs = {}

        # "public-facing" attributes:
        self.name = name
        self.requires_weights = name.startswith('weighted')

        # "internal" attribute: (may change at any time)
        self._internal_kwargs = kwargs

    def _kwargs(self):
        """Not for public consumption -- we may change this at any time"""
        return self._internal_kwargs

    def get_dset_props(self, dist_bin_edges):
        return _get_dset_props_helper(self, dist_bin_edges)

    def validate_rslt(self, rslt, dist_bin_edges, *, max_total_count=None):
        _validate_basic_quan_props(self, rslt, dist_bin_edges)

        # do some extra validation
        counts_or_weights_key = self.get_dset_props(dist_bin_edges)[0]
        _validate_counts_or_weights(self, rslt, counts_or_weights_key, max_total_count)

    def postprocess_rslt(self, rslt):
        if len(rslt) == 0:
            return None
        elif self.name in ["variance", "omoment1_var"]:
            # legacy behavior where we applied bessel's correction. technically, the
            # result produced in rslt['variance'] by the core C++ sf function is really
            # variance*counts

            rslt_dict = rslt
            w = (rslt_dict['counts'] > 1)
            rslt_dict['variance'][w] /= (rslt_dict['counts'][w] - 1)
            rslt_dict['variance'][~w] = 0.0

            w_mask = (rslt_dict['counts']  == 0)
            for k,v in rslt_dict.items():
                if k != 'counts':
                    v[w_mask] = np.nan
        else:
            _postprocess(self, rslt)

def get_statconf(statistic, kwargs):
    return StatConf(statistic, kwargs)
