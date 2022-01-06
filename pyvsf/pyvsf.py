from collections import OrderedDict
import ctypes
import os.path


import numpy as np
from more_itertools import zip_equal

from ._kernels import get_kernel

# get the directory of the current file 
_dir_of_cur_file = os.path.dirname(os.path.abspath(__file__))
# get the expected location of the shared library
_lib_path = os.path.join(_dir_of_cur_file, '../src/libvsf.so')

# confirm that the shared library exists
if not os.path.isfile(_lib_path):
    raise RuntimeError(f"libvsf.so wasn't found at: '{_lib_path}'")

# now actually load in the shared library
_lib = ctypes.CDLL(_lib_path)

_double_ptr = ctypes.POINTER(ctypes.c_double)

class POINTPROPS(ctypes.Structure):
    _fields_ = [("positions", _double_ptr),
                ("velocities", _double_ptr),
                ("n_points", ctypes.c_size_t),
                ("n_spatial_dims", ctypes.c_size_t)]

    @staticmethod
    def construct(pos, vel, dtype = np.float64, allow_null_pair = False):
        if allow_null_pair and (pos is None) and (vel is None):
            return POINTPROPS(None, None, n_points = 0, n_spatial_dims = 0)
        elif (pos is None) or (vel is None):
            raise ValueError("pos and vel must not be None")

        pos_arr = np.asarray(pos, dtype = dtype, order = 'C')
        vel_arr = np.asarray(vel, dtype = dtype, order = 'C')
        assert pos_arr.ndim == 2
        assert vel_arr.ndim == 2

        assert pos_arr.shape == vel_arr.shape
        n_spatial_dims = int(pos_arr.shape[0])
        n_points = int(pos_arr.shape[1])

        return POINTPROPS(positions = pos_arr.ctypes.data_as(_double_ptr),
                          velocities = vel_arr.ctypes.data_as(_double_ptr),
                          n_points = n_points,
                          n_spatial_dims = n_spatial_dims)

class STATLISTITEM(ctypes.Structure):
    _fields_ = [("statistic", ctypes.c_char_p),
                ("arg_ptr", ctypes.c_void_p)]

_STATLISTITEM_ptr = ctypes.POINTER(STATLISTITEM)

class StatList:
    MAX_CAPACITY = 4
    STATLISTITEM_ArrayType = STATLISTITEM * 4

    def __init__(self):
        self.capacity = self.MAX_CAPACITY
        self.length = 0
        self._data = self.STATLISTITEM_ArrayType()
        for i in range(self.capacity):
            self._data[i].statistic = ctypes.c_char_p(None)
            self._data[i].arg_ptr = ctypes.c_void_p(None)

        self._attached_objects = []

    def _attach_object(self, obj):
        """
        Store a reference to obj.

        This is a crude approach for making sure that arbitrary objects have 
        lifetimes that are at least as long as that of self
        """
        if obj not in self._attached_objects:
            self._attached_objects.append(obj)

    def append(self,statistic_name_ptr, arg_struct_ptr = None):
        assert (self.length + 1) <= self.capacity
        new_ind = self.length
        self.length+=1

        if isinstance(statistic_name_ptr, ctypes.Array):
            self._attach_object(statistic_name_ptr) # extra safety
            self._data[new_ind].statistic = ctypes.cast(statistic_name_ptr,
                                                        ctypes.c_char_p)
        else:
            self._data[new_ind].statistic = statistic_name_ptr

        if arg_struct_ptr is None:
            self._data[new_ind].arg_ptr = ctypes.c_void_p(None)
        else:
            self._data[new_ind].arg_ptr = arg_struct_ptr

    def __len__(self):
        return self.length

    def get_STATLISTITEM_ptr(self):
        return ctypes.cast(self._data, _STATLISTITEM_ptr)

    def __str__(self):
        elements = []
        for i in range(self.length):
            elements.append('{' + str(self._data[i].statistic) + ',' +
                            str(self._data[i].arg_ptr) + '}')
        return '[' + ','.join(elements) + ']'

class HISTBINS(ctypes.Structure):
    _fields_ = [("bin_edges", _double_ptr),
                ("n_bins", ctypes.c_size_t)]

    @staticmethod
    def construct(bin_edges, arg_name = None):
        bin_edges = np.asarray(bin_edges, dtype = np.float64, order = 'C')
        assert _verify_bin_edges(bin_edges)
        n_bins = int(bin_edges.size - 1)

        return HISTBINS(bin_edges = bin_edges.ctypes.data_as(_double_ptr),
                        n_bins = n_bins)
_HISTBINS_ptr = ctypes.POINTER(HISTBINS)

_ptr_to_double_ptr = ctypes.POINTER(_double_ptr)

# define the argument types
_lib.calc_vsf_props.argtypes = [
    POINTPROPS, POINTPROPS,
    _STATLISTITEM_ptr, ctypes.c_size_t,
    np.ctypeslib.ndpointer(dtype = np.float64, ndim = 1,
                           flags = 'C_CONTIGUOUS'),
    ctypes.c_size_t,
    np.ctypeslib.ndpointer(dtype = np.float64, ndim = 1,
                           flags = ['C_CONTIGUOUS', 'WRITEABLE']),
    np.ctypeslib.ndpointer(dtype = np.int64, ndim = 1,
                           flags = ['C_CONTIGUOUS', 'WRITEABLE'])
]
_lib.calc_vsf_props.restype = ctypes.c_bool

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

def _process_statistic_args(statistic_l, kwargs_l, dist_bin_edges):
    """
    Construct the appropriate instance of StatList as well as information about
    the output data
    """

    # it's important that we retain order!
    int64_quans = OrderedDict()
    float64_quans = OrderedDict()

    stat_list = StatList()

    stat_kw_pairs = list(zip_equal(statistic_l, kwargs_l))

    # it's important that the statistics_l are ordered in alphabetical order
    # so that the stat_list is initialized in alphabetical order
    stat_kw_pairs.sort(key = lambda pair: pair[0])

    for stat_name, stat_kw in stat_kw_pairs:
        # load kernel object, which stores metadata
        kernel = get_kernel(stat_name)
        if kernel.non_vsf_func is not None:
            raise ValueError(f"'{stat_name}' can't be computed by vsf_props")

        # first, look at quantities associated with stat_name
        prop_l = kernel.get_dset_props(dist_bin_edges, kwargs = stat_kw)
        for quan_name, dtype, shape in prop_l:
            key = (stat_name, quan_name)
            assert (key not in int64_quans) and (key not in float64_quans)
            if dtype == np.int64:
                int64_quans[key] = shape
            elif dtype == np.float64:
                float64_quans[key] = shape
            else:
                raise ValueError(f"can't handle datatype: {dtype}")

        # now, appropriately update StatList
        # kernel.get_dset_props would have raised an error if stat_kw had the
        # wrong size
        c_stat_name_buffer = ctypes.create_string_buffer(stat_name.encode())
        # attach c_stat_name_buffer to stat_list so it isn't garbage collected
        # during stat_list's lifetime
        stat_list._attach_object(c_stat_name_buffer)

        if len(stat_kw) == 0:
            stat_list.append(statistic_name_ptr = c_stat_name_buffer,
                             arg_struct_ptr = None)
        elif stat_name == 'histogram':
            assert list(stat_kw) == ['val_bin_edges']
            val_bin_edges = np.asanyarray(stat_kw['val_bin_edges'],
                                          dtype = np.float64)
            if not _verify_bin_edges(val_bin_edges):
                raise ValueError(
                    'kwargs["val_bin_edges"] must be a 1D monotonically '
                    'increasing array with 2 or more values'
                )
            val_bins_struct = HISTBINS.construct(val_bin_edges)
            val_bins_ptr = _HISTBINS_ptr(val_bins_struct)

            accum_arg_ptr = ctypes.cast(val_bins_ptr, ctypes.c_void_p)
            stat_list.append(statistic_name_ptr = c_stat_name_buffer,
                             arg_struct_ptr = accum_arg_ptr)
            stat_list._attach_object(val_bins_struct)
        else:
            raise RuntimeError(f"There's no support for adding '{stat_name}' "
                               "to stat_list")

    return stat_list, VSFPropsRsltContainer(int64_quans = int64_quans,
                                            float64_quans = float64_quans)

def vsf_props(pos_a, pos_b, vel_a, vel_b, dist_bin_edges,
              statistic = 'variance', kwargs = {}):
    """
    Calculates properties pertaining to the velocity structure function for 
    pairs of points.

    If you set both ``pos_b`` and ``vel_b`` to ``None`` then the velocity 
    structure properties will only be computed for unique pairs of the points
    specified by ``pos_a`` and ``vel_a``

    Parameters
    ----------
    pos_a, pos_b : array_like
        2D arrays holding the positions of each point. Axis 0 should be the 
        number of spatial dimensions must be consistent for each array. Axis 1
        can be different for each array
    vel_a, vel_b : array_like
        2D arrays holding the velocities at each point. The shape of ``vel_a`` 
        should match ``pos_a`` and the shape of ``vel_b`` should match
        ``pos_b``.
    dist_bin_edges : array_like
        1D array of monotonically increasing values that represent edges for 
        distance bins. A distance ``x`` lies in bin ``i`` if it lies in the 
        interval ``dist_bin_edges[i] <= x < dist_bin_edges[i+1]``.
    statistic: string, optional
        The name of the statistic to compute. Default is variance.
    kwargs: dict,optional
        Keyword arguments for computing different statistics. This should be 
        empty for most cases.

    Notes
    -----
    Currently allowed values for statistic include: 'mean', 'variance', and
    'histogram'.

    When statistic == 'histogram', this constructs a 2D histogram. The bin 
    edges along axis 0 are given by the dist_bin_edges argument. The velocity
    differences are binned along axis 1. This function checks the 
    'val_bin_edges' entry from kwargs for a 1D monotonic array that specifies 
    the bin edges along axis 1.
    """

    points_a = POINTPROPS.construct(pos_a, vel_a, dtype = np.float64,
                                    allow_null_pair = False)
    points_b = POINTPROPS.construct(pos_b, vel_b, dtype = np.float64,
                                    allow_null_pair = True)

    if pos_b is None:
        assert points_a.n_points > 1
    else:
        assert points_a.n_spatial_dims == points_b.n_spatial_dims

    if points_a.n_spatial_dims != 3:
        raise NotImplementedError(
            "vsf_props currently only has support for computing velocity "
            "structure function properties for sets of points with 3 spatial "
            "dimensions"
        )

    dist_bin_edges = np.asanyarray(dist_bin_edges, dtype = np.float64)
    if not _verify_bin_edges(dist_bin_edges):
        raise ValueError(
            'dist_bin_edges must be a 1D monotonically increasing array with '
            '2 or more values'
        )
    ndist_bins = dist_bin_edges.size - 1

    if isinstance(statistic, str):
        assert isinstance(kwargs, dict)
        statistic_l = [statistic]
        kwargs_l = [kwargs]
        single_stat = True
    else:
        statistic_l = statistic
        kwargs_l = kwargs
        single_stat = False

    stat_list, rslt_container = _process_statistic_args(statistic_l, kwargs_l,
                                                        dist_bin_edges)

    # now actually call the function
    success = _lib.calc_vsf_props(
        points_a, points_b,
        stat_list.get_STATLISTITEM_ptr(), len(stat_list),
        dist_bin_edges, ndist_bins,
        rslt_container.get_flt_vals_arr(),
        rslt_container.get_i64_vals_arr()
    )

    assert success

    out = []
    for stat_name in statistic_l:
        val_dict = rslt_container.extract_statistic_dict(stat_name)
        if stat_name in ['mean', 'variance']:
            w_mask = (val_dict['counts']  == 0)
            for k,v in val_dict.items():
                if k == 'counts':
                    continue
                else:
                    v[w_mask] = np.nan
        out.append(val_dict)
    if single_stat:
        return out[0]
    else:
        return out
