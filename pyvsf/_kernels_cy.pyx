from copy import deepcopy
import numpy as np

from libc.stdint cimport int64_t
from libc.stddef cimport size_t

from cpython.version cimport PY_MAJOR_VERSION

from ._ArrayDict_cy import ArrayMap

"""
The following duplicates some code from the VarAccum c++ class
"""

cdef:
    struct _VarAccum:
        int64_t count
        double mean
        double cur_M2

    void _add_single_entry_to_VarAccum(_VarAccum *accum, double val) nogil:
        if accum is NULL:
            return
        accum.count += 1
        cdef double val_minus_last_mean
        val_minus_last_mean = val - accum.mean;
        accum.mean += (val_minus_last_mean)/accum.count;
        cdef double val_minus_cur_mean = val - accum.mean;
        accum.cur_M2 += val_minus_last_mean * val_minus_cur_mean;

    _VarAccum combine_pair(_VarAccum a, _VarAccum b) nogil:
        cdef _VarAccum out
        cdef double delta, delta2

        if a.count == 0:
            out = b
        elif b.count == 0:
            out = a
        elif a.count == 1:   # equivalent to adding a single entry to b
            out = b
            _add_single_entry_to_VarAccum(&out, a.mean)
        elif b.count == 1:   # equivalent to adding a single entry to a
            out = a
            _add_single_entry_to_VarAccum(&out, b.mean)
        else:                # general case
            out.count = a.count + b.count
            delta = b.mean - a.mean
            delta2 = delta * delta
            out.cur_M2 = (
                a.cur_M2 + b.cur_M2 + delta2 * (a.count * b.count / out.count)
            )

            if True:
                # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
                # suggests that approach is more stable when the values of
                # a.count and b.count are approximately the same and large
                out.mean = (a.count*a.mean + b.count*b.mean)/out.count
            else:
                # in the other, limit, the following may be more stable
                out.mean = a.mean + (delta * b.mean / out.count)
        return out

def _consolidate_variance(rslts):

    def _load_arrays(rslt):
        counts = rslt['counts'].astype(np.int64, copy = True)

        mean = rslt['mean'].astype(np.float64, copy = True)
        # wherever counts is equal to 0, mean should be 0.0
        mean[counts == 0] = 0.0

        M2 = np.empty(mean.shape, dtype = np.float64)
        # wherever counts is equal to 0 or 1, cur_M2 should be 0.0
        w = counts > 1
        M2[~w] = 0.0
        M2[w] = rslt['variance'][w] * (counts[w] - 1.0)

        return counts, mean, M2

    if len(rslts) == 0:
        raise RuntimeError()
    elif len(rslts) == 1:
        return deepcopy(rslts[0])

    accum_counts, accum_mean, accum_M2 = None, None, None

    cdef Py_ssize_t i
    cdef _VarAccum a,b,tmp

    for rslt in rslts:
        if len(rslt) == 0:
            continue
        cur_counts, cur_mean, cur_M2 = _load_arrays(rslt)
        if accum_counts is None:
            accum_counts, accum_mean, accum_M2 = _load_arrays(rslt)
        else:
            for i in range(len(accum_counts)):
                a.count = accum_counts[i]
                a.mean = accum_mean[i]
                a.cur_M2 = accum_M2[i]

                b.count = cur_counts[i]
                b.mean = cur_mean[i]
                b.cur_M2 = cur_M2[i]

                tmp = combine_pair(a, b)
                accum_counts[i] = tmp.count
                accum_mean[i] = tmp.mean
                accum_M2[i] = tmp.cur_M2

    if accum_counts is None:
        return {}

    accum_mean[accum_counts == 0] = np.nan
    variance = np.empty_like(accum_M2)
    w = accum_counts > 1
    variance[~w] = np.nan
    variance[w] = accum_M2[w] / (accum_counts[w] - 1.0)
    return {'counts' : accum_counts, 'mean' : accum_mean, 'variance' : variance}

def _validate_basic_quan_props(kernel, rslt, dist_bin_edges, kwargs = {}):
    quan_props = kernel.get_dset_props(dist_bin_edges, kwargs)
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

def _allocate_unintialized_rslt_dict(kernel, dist_bin_edges, kwargs = {}):
    quan_props = kernel.get_dset_props(dist_bin_edges, kwargs = kwargs)
    out = {}
    for name, dtype, shape in quan_props:
        out[name] = np.empty(shape = shape, dtype = dtype)
    return out

class Variance:
    name = "variance"
    output_keys = ('counts', 'mean', 'variance')
    commutative_consolidate = False
    operate_on_pairs = True
    non_vsf_func = None

    @classmethod
    def get_extra_fields(cls, kwargs = {}):
        return None

    @classmethod
    def consolidate_stats(cls, *rslts):
        return _consolidate_variance(rslts)

    @classmethod
    def get_dset_props(cls, dist_bin_edges, kwargs = {}):
        assert kwargs == {}
        assert np.size(dist_bin_edges) and np.ndim(dist_bin_edges) == 1
        return [('counts',   np.int64,   (np.size(dist_bin_edges) - 1,)),
                ('mean',     np.float64, (np.size(dist_bin_edges) - 1,)),
                ('variance', np.float64, (np.size(dist_bin_edges) - 1,))]

    @classmethod
    def validate_rslt(cls, rslt, dist_bin_edges, kwargs = {}):
        _validate_basic_quan_props(cls, rslt, dist_bin_edges, kwargs)

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, kwargs = {}):
        # basically create a result object for a dataset that didn't have any
        # pairs at all
        rslt = _allocate_unintialized_rslt_dict(cls, dist_bin_edges, kwargs)
        for k in rslt.keys():
            if k == 'counts':
                rslt['counts'][...] = 0
            else:
                rslt[k][...] = np.nan
        return rslt


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
    ctypedef struct BinSpecification:
        double* bin_edges
        size_t n_bins

    ctypedef struct StatListItem:
        char* statistic
        void* arg_ptr


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
        return self.kernel.get_dset_props(self.dist_bin_edges,
                                          kwargs = self.kwargs)

    def _purge_values(self):
        # come up with some zero-initialized values
        tmp = self.kernel.zero_initialize_rslt(self.dist_bin_edges,
                                               self.kwargs)

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
    if kernel.name == 'histogram':
        return SFConsolidator(dist_bin_edges, kernel, kwargs)
    return PyConsolidator(kernel)

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
    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, kwargs = {}):
        # basically create a result object for a dataset that didn't have any
        # pairs at all
        rslt = _allocate_unintialized_rslt_dict(cls, dist_bin_edges, kwargs)
        for k in rslt.keys():
            rslt[k][...] = 0
        return rslt
