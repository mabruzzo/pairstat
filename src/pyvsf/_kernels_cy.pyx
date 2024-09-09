from collections import OrderedDict
from collections.abc import Sequence
import numpy as np
import warnings

from libc.stdint cimport int64_t, uintptr_t
from libc.stddef cimport size_t

from cpython.version cimport PY_MAJOR_VERSION

from ._ArrayDict_cy import ArrayMap

#==============================================================================
# In the first chunk of this file, we define an interface for calc_vsf_props
# - before it was written in cython, the interface was previously written using
#   the ctypes module
# - when I rewrote it, I largely kept the same general code structure
# - with that in mind, the code structure (and readability) could definitely be
#   improved
#==============================================================================

def _verify_bin_edges(bin_edges):
    nbins = bin_edges.size - 1
    if bin_edges.ndim != 1:
        return False
    elif nbins <= 0:
        return False
    elif not (bin_edges[1:] > bin_edges[:-1]).all():
        return False
    else:
        return True

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
                        const char * pairwise_op,
                        const StatListItem* stat_list, size_t stat_list_len,
                        const double *bin_edges, size_t nbins,
                        const ParallelSpec parallel_spec,
                        double *out_flt_vals, int64_t *out_i64_vals)

    bint cxx_compiled_with_openmp "compiled_with_openmp"()

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
    # -> these first 2 checks may be redundant with np.array(..., order = 'C')
    assert pos_arr.strides[1] == pos_arr.itemsize
    assert val_arr.strides[-1] == val_arr.itemsize

    # in the future, consider relaxing the following conditions (to maybe 
    # facillitate better data alignment)
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



cdef class _WrappedVoidPtr: # this is just a helper class
    cdef void* ptr
    def __cinit__(self):
        self.ptr = NULL

cdef class PyBinSpecification:
    cdef BinSpecification c_bin_spec # wrapped c++ instance
    cdef object bin_edges # numpy array that owns the pointer stored in the
                          # wrapped struct

    def __cinit__(self, bin_edges):
        self.bin_edges = np.asarray(bin_edges, dtype = np.float64, order = 'C')
        if not self.bin_edges.flags['C_CONTIGUOUS']:
            self.bin_edges = np.ascontiguousarray(self.bin_edges)

        assert _verify_bin_edges(self.bin_edges)
        n_bins = int(self.bin_edges.size - 1)

        cdef double[::1] bin_edges_memview = self.bin_edges
        self.c_bin_spec.bin_edges = &bin_edges_memview[0]
        self.c_bin_spec.n_bins = n_bins

    def wrapped_void_ptr(self):
        cdef _WrappedVoidPtr out = _WrappedVoidPtr()
        out.ptr = <void *>(&(self.c_bin_spec))
        return out

cdef enum:
    _MAX_STATLIST_CAPACITY = 4

cdef class StatList:
    cdef StatListItem[_MAX_STATLIST_CAPACITY] data

    # current length (less than or equal to _MAX_STATLIST_CAPACITY)
    cdef int length

    # the c-string stored in data[i].statistic is a pointer to the buffer
    # of the Python byte string stored in self._py_byte_strs (this is supported
    # by cython magic)
    # -> an important reason for this attributes existence is that it ensures
    #    that the lifetime of the contained strings are consistent with the
    #    lifetime of the rest of the object
    # -> the mechanism for extracting the reference to this string is
    #    handled by cython magic
    cdef object _py_byte_strs

    # data[i].arg_ptr is either NULL or a pointer. In cases where its not NULL,
    # it is a pointer to a value wrapped by the extension-type held in
    # self._attached_storage[i].
    cdef object _arg_storage

    def __cinit__(self):
        self.length = 0
        for i in range(_MAX_STATLIST_CAPACITY):
            self.data[i].statistic = NULL
            self.data[i].arg_ptr = NULL

        self._py_byte_strs = [None for i in range(_MAX_STATLIST_CAPACITY)]
        self._arg_storage = [None for i in range(_MAX_STATLIST_CAPACITY)]

    def append(self, statistic_name, statistic_arg = None):
        assert (self.length + 1) <= _MAX_STATLIST_CAPACITY
        ind = self.length
        self.length+=1

        # handle the statistic name (convert it to bytes instance)
        if isinstance(statistic_name, str):
            statistic_name = statistic_name.encode('ascii')
        elif isinstance(statistic_name, bytearray):
            statistic_name = bytes(statistic_name)
        elif not isinstance(statistic_name, bytes):
            raise ValueError("statistic_name must be coercable to bytes")
        self._py_byte_strs[ind] = statistic_name

        # we rely on cython magic to get a pointer to the byte buffer of the
        # Python byte string
        cdef char* c_stat_name = <bytes>(self._py_byte_strs[ind])

        cdef void* arg_ptr = NULL
        self._arg_storage[ind] = statistic_arg
        if self._arg_storage[ind] is not None:
            ptr_wrapper = self._arg_storage[ind].wrapped_void_ptr()
            arg_ptr = (<_WrappedVoidPtr?>(ptr_wrapper)).ptr

        self.data[ind].statistic = c_stat_name
        self.data[ind].arg_ptr = arg_ptr

    def __len__(self):
        return self.length

    def __str__(self):
        cdef uintptr_t tmp
        elements = []
        for i in range(self.length):
            if self.data[i].arg_ptr == NULL:
                ptr_str = 'NULL'
            else:
                tmp = <uintptr_t>(self.data[i].arg_ptr)
                ptr_str = 'ptr(' + hex(int(tmp)) + ')'

            elements.append('{' + self._py_byte_strs[i].decode('ascii') + ',' +
                            ptr_str + '}')
        return '[' + ','.join(elements) + ']'


class VSFPropsRsltContainer:
    def __init__(self, int64_quans, float64_quans):
        duplicates = set(int64_quans.keys()).intersection(float64_quans.keys())
        assert len(duplicates) == 0

        def _parse_input_dict(input_dict):
            total_length = 0
            access_dict = {}
            for key, subarr_shape in input_dict.items():
                subarr_size = np.prod(subarr_shape)
                subarr_idx = slice(total_length, total_length + subarr_size)
                access_dict[key] = (subarr_idx, subarr_shape)
                total_length += subarr_size
            return access_dict, total_length

        self.int64_access_dict,   int64_len   = _parse_input_dict(int64_quans)
        self.float64_access_dict, float64_len = _parse_input_dict(float64_quans)

        self.int64_arr   = np.empty((int64_len,),   dtype = np.int64  )
        self.float64_arr = np.empty((float64_len,), dtype = np.float64)

    @staticmethod
    def _get(key, access_dict, arr):
        idx, out_shape = access_dict[key]
        out = arr[idx]
        out.shape = out_shape # ensures we don't make a copy
        return out

    def __getitem__(self,key):
        try:
            return self._get(key, self.float64_access_dict, self.float64_arr)
        except KeyError:
            try:
                return self._get(key, self.int64_access_dict, self.int64_arr)
            except KeyError:
                raise KeyError(key) from None

    def extract_statistic_dict(self, statistic_name):
        out = {}

        def _extract(access_dict, arr):
            for (stat,quan), v in access_dict.items():
                if stat == statistic_name:
                    out[quan] = self._get((stat,quan), access_dict, arr)

        _extract(self.int64_access_dict,   self.int64_arr  )
        _extract(self.float64_access_dict, self.float64_arr)

        if len(out) == 0:
            raise ValueError(f"there's no statistic called '{statistic_name}'")
        return out

    def get_flt_vals_arr(self):
        return self.float64_arr

    def get_i64_vals_arr(self):
        return self.int64_arr

def _process_statistic_args(statconf_l, dist_bin_edges):
    """
    Construct the appropriate instance of StatList as well as information about
    the output data
    """

    # it's important that we retain order!
    int64_quans = OrderedDict()
    float64_quans = OrderedDict()

    stat_list = StatList()

    # it's important that we consider the entries of stat_kw_pairs in
    # alphabetical order of the statistic names so that the stat_list entries
    # are also initialized in alphabetical order
    for statconf in sorted(statconf_l, key = lambda statconf: statconf.name):

        # first, look at output quantities associated with stat_name
        prop_l = statconf.get_dset_props(dist_bin_edges)
        for quan_name, dtype, shape in prop_l:
            key = (statconf.name, quan_name)
            if (key in int64_quans) or (key in float64_quans):
                raise ValueError(f"{key} already appears as an output for a "
                                 "different statistic")
            if dtype == np.int64:
                int64_quans[key] = shape
            elif dtype == np.float64:
                float64_quans[key] = shape
            else:
                raise ValueError(f"can't handle datatype: {dtype}")

        # now, update StatList
        if statconf.name in ['histogram', 'weightedhistogram']:
            val_bin_edges = np.asanyarray(statconf.val_bin_edges, dtype = 'f8')
            if not _verify_bin_edges(val_bin_edges):
                raise ValueError(
                    'kwargs["val_bin_edges"] must be a 1D monotonically '
                    'increasing array with 2 or more values'
                )
            val_bin_spec = PyBinSpecification(bin_edges = val_bin_edges)
            stat_list.append(statconf.name, val_bin_spec)
        else:
            stat_list.append(statistic_name = statconf.name, statistic_arg = None)

    return stat_list, VSFPropsRsltContainer(int64_quans = int64_quans,
                                            float64_quans = float64_quans)

def _coerce_stat_list(arg):
    argname = 'stat_kw_pairs'


    def is_statconf(elem):
        return isinstance(elem, _SF_STATCONF_TUPLE)

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


def _core_pairwise_work(pos_a, pos_b, val_a, val_b, dist_bin_edges,
                        weights_a = None, weights_b = None,
                        pairwise_op = "sf",
                        stat_kw_pairs = [('variance', {})],
                        nproc = 1, force_sequential = False,
                        postprocess_stat = True):
    statconf_l = _coerce_stat_list(stat_kw_pairs)

    val_is_vector = (pairwise_op == "sf")
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
    elif (weights_a is not None) and (pairwise_op != "sf"):
        raise ValueError("you can't provide weights_a unless you are using "
                         "pariwise_op == 'sf'")
    elif (points_a.has_non_positive_weights() or
          points_b.has_non_positive_weights()):
        raise ValueError("you can't provide non-positive weights")

    # check if any statistics requre weights
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
    stat_list, rslt_container = _process_statistic_args(statconf_l, dist_bin_edges)

    cdef ParallelSpec parallel_spec
    parallel_spec.nproc = nproc
    parallel_spec.force_sequential = force_sequential

    # setup the pointers to the output buffers
    cdef double* out_flt_vals = NULL
    cdef double[::1] out_flt_memview
    if rslt_container.get_flt_vals_arr().size > 0:
        out_flt_memview = rslt_container.get_flt_vals_arr()
        out_flt_vals = &(out_flt_memview[0])

    cdef int64_t* out_i64_vals = NULL
    cdef int64_t[::1] out_i64_memview
    if rslt_container.get_i64_vals_arr().size > 0:
        out_i64_memview = rslt_container.get_i64_vals_arr()
        out_i64_vals = &(out_i64_memview[0])

    cdef bytes casted_pairwise_op = pairwise_op.encode("ASCII")

    cdef const char* c_pairwise_op = casted_pairwise_op

    cdef bint success = calc_vsf_props(
        points_a = points_a.c_points, points_b = points_b.c_points,
        pairwise_op = c_pairwise_op,
        stat_list = (<StatList?>stat_list).data,
        stat_list_len = len(stat_list),
        bin_edges = &(bin_edges_view[0]), nbins = ndist_bins,
        parallel_spec = parallel_spec, 
        out_flt_vals = out_flt_vals, out_i64_vals = out_i64_vals
    )

    if not success:
        raise RuntimeError("Something went wrong while in calc_vsf_props")

    out = []
    for statconf in statconf_l:
        val_dict = rslt_container.extract_statistic_dict(statconf.name)

        if postprocess_stat:
            statconf.postprocess_rslt(val_dict)
        out.append(val_dict)

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
        ``np.shape(pos_b)[1]``. It is an error to specify weights when no
        statistics will be computed that use them.
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
    Currently recognized statistic names include:
        - 'mean': calculate the 1st order structure function.
        - 'variance': calculate the 1st and 2nd order structure functions
        - 'histogram': this constructs a 2D histogram. The bin edges along
          axis 0 are given by the `dist_bin_edges` argument. The
          magnitudes of the vector differences are binned along axis 1.
          The 'val_bin_edges' keyword must be specified alongside this
          statistic name (to specify the bin edges along axis 1). It
          should be associated with a 1D monotonic array.
        - 'weightedmean': just like 'mean', but weights are used
        - 'weightedhistogram': just like 'histogram', but weights are used
    """


    # do some messy work to help us deprecate vel_a and vel_b

    # Step 1: we do some basic preperation
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

    return _core_pairwise_work(
        pos_a = pos_a, pos_b = pos_b, val_a = _val_a, val_b = _val_b,
        dist_bin_edges = dist_bin_edges, 
        weights_a = weights_a, weights_b = weights_b, pairwise_op = "sf",
        stat_kw_pairs = stat_kw_pairs, nproc = nproc,
        force_sequential = force_sequential,
        postprocess_stat = postprocess_stat)


def twopoint_correlation(pos_a, pos_b, val_a, val_b, dist_bin_edges,
                         *, stat_kw_pairs = [('mean', {})],
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
        1D arrays holding the velocities at each point. The shape of ``vel_a`` 
        should match ``pos_a`` and the shape of ``vel_b`` should match
        ``pos_b``.
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
        - 'mean': the typical correlation function
        - 'variance': the variance of the products of the pairs of scalar
          values are computed for all pairs of values in a given bin.
        - 'histogram': this constructs a 2D histogram. The bin edges along
          axis 0 are given by the `dist_bin_edges` argument. The products
          of the pairs of scalar values are binned along axis 1. The
          'val_bin_edges' keyword must be specified alongside this
          statistic name (to specify the bin edges along axis 1). It
          should be associated with a 1D monotonic array.
    """

    return _core_pairwise_work(
        pos_a = pos_a, pos_b = pos_b, val_a = val_a, val_b = val_b,
        dist_bin_edges = dist_bin_edges, weights_a = None, weights_b = None,
        pairwise_op = "correlate", stat_kw_pairs = stat_kw_pairs, nproc = nproc,
        force_sequential = force_sequential, postprocess_stat = True)

#==============================================================================
# It's been a long time since I've looked at the next chunk of code, but I
# think it could be integrated with the above chunk to some degree
# - the following section is related to defining "Kernels" for
#   structure-function statistics
#==============================================================================

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


cdef void* _construct_accum_handle(size_t num_dist_bins, object name,
                                   object quan_bin_edges_arr = None):
    assert PY_MAJOR_VERSION >= 3

    cdef bytes coerced_name_str

    if isinstance(name, str):
        coerced_name_str = name.encode('ASCII')
    elif isinstance(name, (bytes, bytearray)):
        coerced_name_str = bytes(name)
    else:
        raise ValueError("name must have the type: str, bytes, or bytearray")

    # lifetime of c_name_str is tied to coerced_name_str
    cdef char* c_name_str = coerced_name_str

    cdef StatListItem list_entry
    list_entry.statistic = c_name_str

    
    cdef BinSpecification bin_spec    
    if quan_bin_edges_arr is not None:
        # lifetime of bin_spec is tied to quan_bin_edges_arr
        bin_spec = _build_BinSpecification(quan_bin_edges_arr, True)
        list_entry.arg_ptr = <void*>(&bin_spec)
    else:
        list_entry.arg_ptr = NULL

    return accumhandle_create(&list_entry, 1, num_dist_bins)

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
    cdef object kernel
    cdef object kwargs
    cdef object dist_bin_edges

    def __cinit__(self, object dist_bin_edges, object kernel, object kwargs):
        cdef object name = kernel.name
        cdef object val_bin_edges = None
        if 'val_bin_edges' in kwargs:
            assert len(kwargs) == 1
            val_bin_edges = kwargs['val_bin_edges']
        else:
            assert len(kwargs) == 0

        cdef size_t num_dist_bins = dist_bin_edges.size - 1
        self.primary_handle = _construct_accum_handle(num_dist_bins, name,
                                                      val_bin_edges)
        self.secondary_handle = _construct_accum_handle(num_dist_bins, name,
                                                        val_bin_edges)
        self.kernel = kernel
        self.kwargs = kwargs
        self.dist_bin_edges = dist_bin_edges

    def __dealloc__(self):
        accumhandle_destroy(self.primary_handle)
        accumhandle_destroy(self.secondary_handle)

    def _get_entry_spec(self):
        return self.kernel.get_dset_props(self.dist_bin_edges)

    def _purge_values(self):
        # come up with some zero-initialized values
        tmp = self.kernel.zero_initialize_rslt(self.dist_bin_edges,
                                               postprocess_rslt = False)

        # convert tmp so that it's an instance of ArrayDict
        assert isinstance(tmp, dict) # sanity check
        tmp = ArrayMap.copy_from_dict(tmp)

        # finally restore the handle from the zero-initialized values
        _restore_handle_from_ArrayMap(self.primary_handle, tmp)

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

class PyConsolidator:
    """ Uses the python method built into the stat kernel
    """

    def __init__(self, kernel): self._kernel = kernel

    def consolidate(self, *rslts):
        return self._kernel.consolidate_stats(*rslts)

def build_consolidater(dist_bin_edges, kernel, kwargs):
    if kernel.non_vsf_func is None:
        return SFConsolidator(dist_bin_edges, kernel, kwargs)
    return PyConsolidator(kernel)


def _validate_basic_quan_props(kernel, rslt, dist_bin_edges):
    quan_props = kernel.get_dset_props(dist_bin_edges)
    assert len(quan_props) == len(rslt)
    for name, dtype, shape in quan_props:
        if name not in quan_props:
            raise ValueError(
                f"The result for the '{kernel.name}' statistic is missing a "
                f"quantity called '{name}'"
            )
        elif rslt[name].dtype != dtype:
            raise ValueError(
                f"the {name} quantity for the {kernel.name} statistic should ",
                f"have a dtype of {dtype}, not of {rslt[name].dtype}"
            )
        elif rslt[name].shape != shape:
            raise ValueError(
                f"the {name} quantity for the {kernel.name} statistic should ",
                f"have a shape of {shape}, not of {rslt[name].shape}"
            )

def _allocate_unintialized_rslt_dict(kernel, dist_bin_edges):
    quan_props = kernel.get_dset_props(dist_bin_edges)
    out = {}
    for name, dtype, shape in quan_props:
        out[name] = np.empty(shape = shape, dtype = dtype)
    return out

def _check_bin_edges_arg(arg, arg_description):
    if np.size(arg) < 2 or np.ndim(arg) != 1:
        raise ValueError(f"The {arg_description} must specify a 1D array with "
                         "2 or more elements")

def _check_dist_bin_edges(dist_bin_edges):
    _check_bin_edges_arg(dist_bin_edges, "'dist_bin_edges' argument")

# define functionality shared by both kinds of histograms
# histograms

def _hist_dset_props(kernel, dist_bin_edges):
    val_bin_edges = kernel.val_bin_edges
    _check_dist_bin_edges(dist_bin_edges)

    assert len(kernel.output_keys) == 1
    n = kernel.output_keys[0]
    if kernel.requires_weights:
        t = np.float64
    else:
        t = np.int64
    return [(n, t, (np.size(dist_bin_edges) - 1, np.size(val_bin_edges) - 1))]


def _validate_hist_results(kernel, rslt, dist_bin_edges, used_points = None):

    _validate_basic_quan_props(kernel, rslt, dist_bin_edges)

    # do some extra validation
    key = kernel.output_keys[0]
    if kernel.requires_weights:
        bin_weights = rslt[key]
        if (bin_weights < 0).any():
            raise ValueError("The histogram can't contain negative weights")
    else:
        hist_counts = rslt[key]
        if (hist_counts < 0).any():
            raise ValueError("The histogram can't contain negative counts")

        if used_points is not None:
            # compute the maximum number of pairs of points (make sure to
            # to compute this with python integers (to avoid overflows)
            max_pairs = int(used_points)*max(int(used_points-1),0)//2

            if max_pairs > np.iinfo(hist_counts.dtype).max:
                n_pairs = sum(int(e) for e in hist_counts)
            else:
                n_pairs = np.sum(hist_counts)
            if n_pairs > max_pairs:
                raise ValueError(
                    f"The dataset made use of {used_points} points. The "
                    f"histogram should hold no more than {max_pairs} pairs of "
                    f"points. In reality, it has {n_pairs} pairs."
                )

def _zero_initialize_hist_rslt(kernel, dist_bin_edges,
                               postprocess_rslt):
    # basically create a result object for a dataset that didn't have any
    # pairs at all
    rslt = _allocate_unintialized_rslt_dict(kernel, dist_bin_edges)
    for k in rslt.keys():
        rslt[k][...] = 0
    if postprocess_rslt:
        kernel.postprocess_rslt(rslt)
    return rslt


class Histogram:
    name = "histogram"
    commutative_consolidate = True
    requires_weights = False

    def __init__(self, kwargs):
        if list(kwargs.keys()) != ['val_bin_edges']:
            raise ValueError("'val_bin_edges' is required as the single kwarg for "
                             "computing histogram-statistics")
        self.val_bin_edges = kwargs['val_bin_edges']
        _check_bin_edges_arg(self.val_bin_edges, "'val_bin_edges' kwarg")
        # historically, the following was a class attribute, but that's unnecessary
        self.output_keys = ('2D_counts',)

    def get_dset_props(self, dist_bin_edges):
        return _hist_dset_props(kernel = self, dist_bin_edges = dist_bin_edges)

    def validate_rslt(self, rslt, dist_bin_edges, used_points = None):
        _validate_hist_results(kernel = self, rslt = rslt,
                               dist_bin_edges = dist_bin_edges,
                               used_points = used_points)

    @classmethod
    def postprocess_rslt(cls, rslt):
        pass # do nothing

    def zero_initialize_rslt(self, dist_bin_edges, postprocess_rslt = True):
        # basically create a result object for a dataset that didn't have any
        # pairs at all
        return _zero_initialize_hist_rslt(kernel = self,
                                          dist_bin_edges = dist_bin_edges,
                                          postprocess_rslt = postprocess_rslt)

class WeightedHistogram:
    name = "weightedhistogram"
    commutative_consolidate = False
    requires_weights = True

    def __init__(self, kwargs):
        if list(kwargs.keys()) != ['val_bin_edges']:
            raise ValueError("'val_bin_edges' is required as the single kwarg for "
                             "computing histogram-statistics")
        self.val_bin_edges = kwargs['val_bin_edges']
        _check_bin_edges_arg(self.val_bin_edges, "'val_bin_edges' kwarg")
        # historically, the following was a class attribute, but that's unnecessary
        self.output_keys = ('2D_weight_sums',)

    def get_dset_props(self, dist_bin_edges):
        return _hist_dset_props(kernel = self, dist_bin_edges = dist_bin_edges)

    def validate_rslt(self, rslt, dist_bin_edges, used_points = None):
        _validate_hist_results(kernel = self, rslt = rslt,
                               dist_bin_edges = dist_bin_edges,
                               used_points = used_points)

    @classmethod
    def postprocess_rslt(cls, rslt):
        pass # do nothing

    def zero_initialize_rslt(self, dist_bin_edges, postprocess_rslt = True):
        # basically create a result object for a dataset that didn't have any
        # pairs at all
        return _zero_initialize_hist_rslt(kernel = self,
                                          dist_bin_edges = dist_bin_edges,
                                          postprocess_rslt = postprocess_rslt)



def _set_empty_count_locs_to_NaN(rslt_dict, key = 'counts'):
    w_mask = (rslt_dict[key]  == 0)
    for k,v in rslt_dict.items():
        if k == key:
            continue
        else:
            v[w_mask] = np.nan

class Variance:
    # technically the result returned by pyvsf.vsf_props for 'variance' when
    # post-processing is disabled is variance*counts.

    name = "variance"
    commutative_consolidate = False
    requires_weights = False

    # the following isn't a required class attribute, it's just a common choice
    output_keys = ('counts', 'mean', 'variance')

    def __init__(self, kwargs):
        if (kwargs is not None) and (len(kwargs) > 0):
            raise ValueError("variance takes no kwargs")

    @classmethod
    def get_dset_props(cls, dist_bin_edges):
        assert np.size(dist_bin_edges) and np.ndim(dist_bin_edges) == 1
        return [('counts',   np.int64,   (np.size(dist_bin_edges) - 1,)),
                ('mean',     np.float64, (np.size(dist_bin_edges) - 1,)),
                ('variance', np.float64, (np.size(dist_bin_edges) - 1,))]

    @classmethod
    def validate_rslt(cls, rslt, dist_bin_edges):
        _validate_basic_quan_props(cls, rslt, dist_bin_edges)

    @classmethod
    def postprocess_rslt(cls, rslt):
        if rslt == {}:
            return
        w = (rslt['counts'] > 1)
        # it may not make any sense to use Bessel's correction
        rslt['variance'][w] /= (rslt['counts'][w] - 1)
        rslt['variance'][~w] = 0.0
        _set_empty_count_locs_to_NaN(rslt)

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, postprocess_rslt = True):
        # basically create a result object for a dataset that didn't have any
        # pairs at all
        rslt = _allocate_unintialized_rslt_dict(cls, dist_bin_edges)
        for k in rslt.keys():
            if k == 'counts':
                rslt['counts'][...] = 0
            else:
                rslt[k][...] = 0
        if postprocess_rslt:
            cls.postprocess_rslt(rslt)
        return rslt

class Mean:
    name = "mean"
    commutative_consolidate = False
    requires_weights = False

    # the following isn't a required class attribute, it's just a common choice
    output_keys = ('counts', 'mean')

    def __init__(self, kwargs):
        if (kwargs is not None) and (len(kwargs) > 0):
            raise ValueError("mean takes no kwargs")


    @classmethod
    def get_dset_props(cls, dist_bin_edges):
        assert np.size(dist_bin_edges) and np.ndim(dist_bin_edges) == 1
        return [('counts',   np.int64,   (np.size(dist_bin_edges) - 1,)),
                ('mean',     np.float64, (np.size(dist_bin_edges) - 1,))]

    @classmethod
    def validate_rslt(cls, rslt, dist_bin_edges):
        _validate_basic_quan_props(cls, rslt, dist_bin_edges)

    @classmethod
    def postprocess_rslt(cls, rslt):
        _set_empty_count_locs_to_NaN(rslt)

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, postprocess_rslt = True):
        raise NotImplementedError()

class WeightedMean:
    name = "weightedmean"
    commutative_consolidate = False
    requires_weights = True

    # the following isn't a required class attribute, it's just a common choice
    output_keys = ('weight_sum', 'mean')

    def __init__(self, kwargs):
        if (kwargs is not None) and (len(kwargs) > 0):
            raise ValueError("weightedmean takes no kwargs")

    @classmethod
    def get_dset_props(cls, dist_bin_edges):
        assert np.size(dist_bin_edges) and np.ndim(dist_bin_edges) == 1
        return [('weight_sum', np.float64, (np.size(dist_bin_edges) - 1,)),
                ('mean',       np.float64, (np.size(dist_bin_edges) - 1,))]

    @classmethod
    def validate_rslt(cls, rslt, dist_bin_edges):
        _validate_basic_quan_props(cls, rslt, dist_bin_edges)

    @classmethod
    def postprocess_rslt(cls, rslt):
        _set_empty_count_locs_to_NaN(rslt, 'weight_sum')

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, postprocess_rslt = True):
        raise NotImplementedError()

class StatConfFactory:
    def __init__(self, itr):
        self._kdict = dict((kernel.name, kernel) for kernel in set(itr))
    def get_kernel(self, statistic, kwargs):
        try:
            return self._kdict[statistic](kwargs)
        except KeyError:
            # the `from None` clause avoids exception chaining
            raise ValueError(f"Unknown Statistic: {statistic}") from None

_SF_STATCONF_TUPLE = (Mean, Variance, Histogram, WeightedMean, WeightedHistogram,)
_SF_STATCONF_REGISTRY = StatConfFactory(_SF_STATCONF_TUPLE)

def get_statconf(statistic, kwargs):
    return _SF_STATCONF_REGISTRY.get_kernel(statistic, kwargs)
