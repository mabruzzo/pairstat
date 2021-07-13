import ctypes
import os.path

import numpy as np

# get the directory of the current file 
_dir_of_cur_file = os.path.dirname(os.path.abspath(__file__))
# get the expected location of the shared library
_lib_path = os.path.join(_dir_of_cur_file, 'libvsf.so')

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



def vsf_props(pos_a, pos_b, vel_a, vel_b, dist_bin_edges,
              statistic = 'variance'):
    """
    Calculates properties pertaining to the velocity structure function for 
    pairs of points.

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
    """

    points_a = POINTPROPS.construct(pos_a, vel_a, dtype = np.float64,
                                    allow_null_pair = False)
    points_b = POINTPROPS.construct(pos_b, vel_b, dtype = np.float64,
                                    allow_null_pair = True)

    if pos_b is not None:
        assert points_a.n_spatial_dims == points_b.n_spatial_dims

    dist_bin_edges = np.asanyarray(dist_bin_edges, dtype = np.float64)
    assert dist_bin_edges.ndim == 1
    nbins = dist_bin_edges.size - 1
    assert nbins > 0
    assert (dist_bin_edges[1:] > dist_bin_edges[:-1]).all()

    
    out_counts = np.empty((nbins,), dtype = np.int64)
    statistic_name = ctypes.create_string_buffer(statistic.encode())

    if statistic == 'mean':
        out_vals = np.empty((nbins,), dtype = np.float64)
    elif statistic == 'variance':
        out_vals = np.empty((2*nbins,), dtype = np.float64)
    else:
        raise RuntimeError('Unknown statistic')

    # now actually call the function
    success = _lib.calc_vsf_props(points_a, points_b, statistic_name,
                                  dist_bin_edges, nbins, out_vals, out_counts)
    assert success

    if statistic == 'mean':
        val_dict = {'mean' : out_vals[:nbins]}
    else:
        val_dict = {'mean' : out_vals[:nbins], 'variance' : out_vals[nbins:]}

    w_mask = (out_counts == 0)
    for k,v in val_dict.items():
        v[w_mask] = np.nan
    return out_counts, val_dict


if __name__ == '__main__':
    x_a = np.arange(6.0).reshape(3,2)
    vel_a = np.arange(-3.0,3.0).reshape(3,2)
    
    x_b = np.arange(6.0,24.0).reshape(3,6)
    vel_b = np.arange(-9.0,9.0).reshape(3,6)
    print(x_a)
    print(vel_a)
    print(x_b)
    print(vel_b)

    bin_edges = np.array([17.0, 21.0, 25.0])
    print(vsf_props(x_a,  x_b, vel_a, vel_b, bin_edges))
