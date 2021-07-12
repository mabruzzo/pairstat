import ctypes
import os.path

import numpy as np

# get the directory of the current file 
_dir_of_cur_file = os.path.dirname(os.path.abspath(__file__))
# get the expected location of the shared library
_lib_path = os.path.join(_dir_of_cur_file, 'test.so')

# confirm that the shared library exists
if not os.path.isfile(_lib_path):
    raise RuntimeError(
        "The test.so shared library wasn't be found at: '{_lib_path}'"
    )

# now actually load in the shared library
_lib = ctypes.CDLL(path)


_int64_ptr = ctypes.POINTER(ctypes.c_int64)
_double_ptr = ctypes.POINTER(ctypes.c_double)
_ptr_to_double_ptr = ctypes.POINTER(_double_ptr)

# define the argument types
_lib.calc_vsf_props.argtypes = [
    _ptr_to_double_ptr, _ptr_to_double_ptr, ctypes.c_size_t,
    _ptr_to_double_ptr, _ptr_to_double_ptr, ctypes.c_size_t,
    ctypes.c_uint8,
    _double_ptr, ctypes.c_size_t,
    _double_ptr, _int64_ptr
]
_lib.calc_vsf_props.restype = ctypes.c_bool


def vsf_props(points_1, vel_1, points_2, vel_2, bin_edges):
    """
    Calculates properties pertaining to the velocity structure function for 
    pairs of points.

    Parameters
    ----------
    points : tuple of ndarray of float, with fixed shape 

    """
    pass
