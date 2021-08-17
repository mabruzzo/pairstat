import ctypes
import os.path

import numpy as np

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


_ptr_to_double_ptr = ctypes.POINTER(_double_ptr)

# define the argument types
_lib.calc_vsf_props.argtypes = [
    POINTPROPS, POINTPROPS, ctypes.c_char_p,
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

    statistic_name = ctypes.create_string_buffer(statistic.encode())

    if statistic == 'histogram':
        assert list(kwargs.keys()) == ['val_bin_edges']
        val_bin_edges = np.asanyarray(kwargs['val_bin_edges'],
                                      dtype = np.float64)
        if not _verify_bin_edges(val_bin_edges):
            raise ValueError(
                'kwargs["dist_bin_edges"] must be a 1D monotonically '
                'increasing array with 2 or more values'
            )

        nval_bins = val_bin_edges.size - 1
        # out_flt_vals isn't expected to have any values. but for simplicity of
        # passing arguments through ctypes, give it a single dummy argument
        out_flt_vals = np.empty((1,), dtype = np.float64)
        out_i64_vals = np.empty((ndist_bins * nval_bins,),
                                dtype = np.int64)
    else:
        assert len(kwargs) == 0
        if statistic == 'mean':
            out_flt_vals = np.empty((ndist_bins,), dtype = np.float64)
            out_i64_vals = np.empty((ndist_bins,), dtype = np.int64)
        elif statistic == 'variance':
            out_flt_vals = np.empty((2*ndist_bins,), dtype = np.float64)
            out_i64_vals = np.empty((ndist_bins,), dtype = np.int64)
        else:
            raise RuntimeError('Unknown statistic')

    # now actually call the function
    success = _lib.calc_vsf_props(points_a, points_b, statistic_name,
                                  dist_bin_edges, ndist_bins, out_flt_vals,
                                  out_i64_vals)
    assert success

    if statistic in ['mean', 'variance']:
        if statistic == 'mean':
            val_dict = {'mean' : out_flt_vals[:ndist_bins]}
        else:
            val_dict = {'mean' : out_flt_vals[:ndist_bins],
                        'variance' : out_flt_vals[ndist_bins:]}

        val_dict['counts'] = out_i64_vals
        w_mask = (val_dict['counts']  == 0)
        for k,v in val_dict.items():
            if k == 'counts':
                continue
            else:
                v[w_mask] = np.nan

    elif statistic == 'histogram':
        val_dict = {'2D_counts' : out_i64_vals}
        val_dict['2D_counts'].shape = (ndist_bins, nval_bins)

    return val_dict
