from copy import deepcopy
import numpy as np

from libc.stdint cimport int64_t
from libc.stddef cimport size_t

from cpython.version cimport PY_MAJOR_VERSION

from ._ArrayDict_cy import ArrayMap

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
        tmp = self.kernel.zero_initialize_rslt(self.dist_bin_edges, self.kwargs,
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



class Histogram:
    name = "histogram"
    output_keys = ('2D_counts',)
    commutative_consolidate = True
    operate_on_pairs = True
    non_vsf_func = None

    @classmethod
    def n_ghost_ax_end(cls):
        return 0
    
    @classmethod
    def get_extra_fields(cls, kwargs = {}):
        return None

    @classmethod
    def consolidate_stats(cls, *rslts):
        raise RuntimeError("THIS SHOULD NOT BE CALLED")

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
    def postprocess_rslt(cls, rslt):
        pass # do nothing

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, kwargs = {},
                             postprocess_rslt = True):
        # basically create a result object for a dataset that didn't have any
        # pairs at all
        rslt = _allocate_unintialized_rslt_dict(cls, dist_bin_edges, kwargs)
        for k in rslt.keys():
            rslt[k][...] = 0
        if postprocess_rslt:
            cls.postprocess_rslt(rslt)
        return rslt


def _set_empty_count_locs_to_NaN(rslt_dict):
    w_mask = (rslt_dict['counts']  == 0)
    for k,v in rslt_dict.items():
        if k == 'counts':
            continue
        else:
            v[w_mask] = np.nan

class Variance:
    # technically the result returned by pyvsf.vsf_props for 'variance' when
    # post-processing is disabled is variance*counts.

    name = "variance"
    output_keys = ('counts', 'mean', 'variance')
    commutative_consolidate = False
    operate_on_pairs = True
    non_vsf_func = None

    @classmethod
    def n_ghost_ax_end(cls):
        return 0

    @classmethod
    def get_extra_fields(cls, kwargs = {}):
        return None

    @classmethod
    def consolidate_stats(cls, *rslts):
        raise RuntimeError("THIS SHOULD NOT BE CALLED")

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
    def postprocess_rslt(cls, rslt):
        if rslt == {}:
            return
        w = (rslt['counts'] > 1)
        # it may not make any sense to use Bessel's correction
        rslt['variance'][w] /= (rslt['counts'][w] - 1)
        rslt['variance'][~w] = 0.0
        _set_empty_count_locs_to_NaN(rslt)

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, kwargs = {},
                             postprocess_rslt = True):
        # basically create a result object for a dataset that didn't have any
        # pairs at all
        rslt = _allocate_unintialized_rslt_dict(cls, dist_bin_edges, kwargs)
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
    output_keys = ('counts', 'mean')
    commutative_consolidate = False
    operate_on_pairs = True
    non_vsf_func = None

    @classmethod
    def n_ghost_ax_end(cls):
        return 0

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
_SF_KERNEL_TUPLE = (Mean, Variance, Histogram)
_SF_KERNEL_REGISTRY = KernelRegistry(_SF_KERNEL_TUPLE)

def get_sf_kernel(statistic):
    return _SF_KERNEL_REGISTRY.get_kernel(statistic)
