# define a map-like object that is used to hold arrays of a predetermined size
from collections.abc import Mapping

import numpy as np

def _array_shape_validation(array_shape):
    if not isinstance(array_shape, tuple):
        raise TypeError("An array_shape should be a tuple specifying")
    elif len(array_shape) == 0:
        raise ValueError("An array_shape can't be empty")
    elif any((not isinstance(e, int)) for e in array_shape):
        raise TypeError("array_shape must only contain integers")
    elif any(e<=0 for e in array_shape):
        raise TypeError("array_shape must only contain positive integers")

class ArrayMapEntrySpec:
    """
    Specifies the entries of an ArrayMap

    This essentially stores a sequence of tuples where each tuple holds:
    (key, dtype, shape, index_start, index_stop)


    Parameters
    ----------
    entry_spec: sequence of tuples of ArrayMapEntrySpec
        This should be an existing ArrayMapEntrySpec or a sequence of tuples
        where each tuple is of the form (key, dtype, array shape)
    """

    def __init__(self, entry_spec):
        if isinstance(entry_spec, ArrayMapEntrySpec):
            self._entries = entry_spec._entries
        else:
            self._entries = self._process_entry_spec(entry_spec)

    def get_dtype_slice(self, target_dtype):
        start_loc = np.inf
        stop_loc = 0
        for key, dtype, shape, index_start, index_stop in self._entries:
            if target_dtype == dtype:
                start_loc = min(index_start, start_loc)
                stop_loc = max(index_stop, stop_loc)
        if np.isinf(start_loc):
            return slice(0,0)
        return slice(start_loc, stop_loc)

    def required_storage_num_uint64(self):
        """
        specify the amount of storage space needed to store data for each entry
        as a multiple of np.dtype(np.uint64).itemsize
        """
        # specifies the amount of storage space (as
        return max(map(lambda e: e[-1], self._entries))

    def num_bytes(self):
        return np.dtype(np.uint64).itemsize * self.num_equivalent_uint64()

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return self._entries.__iter__()

    def _get_entry(self, key):
        # returns None if the entry can't be found
        for entry in self._entries:
            if key == entry[0]:
                return entry
        return None

    def _get_subarray(self, key, array):
        tmp = self._get_entry(key)
        if tmp is None:
            raise KeyError(f"{key}")
        key, dtype, shape, index_start, index_stop = tmp
        view = array[index_start:index_stop]
        view.dtype = dtype
        view.shape = shape
        return view

    def __contains__(self, key):
        return self._get_entry is not None

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._entries == other._entries

    @classmethod
    def _process_entry_spec(cls, entry_spec):
        key_set = set()

        int64_len, uint64_len, float64_len = (0,0,0)
        for key, dtype, array_shape in entry_spec:
            if not isinstance(key, str):
                raise TypeError(
                    "The first entry of each entry_spec tuple should be the "
                    "name of a key (specified as a string)"
                )
            elif key in key_set:
                raise ValueError("multiple keys of the same name were "
                                 "specified")
            else:
                key_set.add(key)

            _array_shape_validation(array_shape)

            dtype = np.dtype(dtype)
            if dtype == np.int64:
                int64_len += np.prod(array_shape)
            elif dtype == np.uint64:
                uint64_len += np.prod(array_shape)
            elif dtype == np.float64:
                float64_len += np.prod(array_shape)
            else:
                raise ValueError(f"can't handle {dtype} dtype")

        # now construct the output
        cur_int64_offset = 0
        cur_uint64_offset = int64_len
        cur_float64_offset = int64_len + uint64_len

        int64_l = []
        uint64_l = []
        float64_l = []

        for key, dtype, array_shape in entry_spec:
            n_entries = np.prod(array_shape)
            dtype = np.dtype(dtype)
            if dtype == np.int64:
                arr_start = cur_int64_offset
                cur_int64_offset += n_entries
            elif dtype == np.uint64:
                arr_start = cur_uint64_offset
                cur_uint64_offset += n_entries
            else:
                arr_start = cur_float64_offset
                cur_float64_offset += n_entries
            tup = (key, dtype, array_shape, arr_start, arr_start + n_entries)
            if dtype == np.int64:
                int64_l.append(tup)
            elif dtype == np.uint64:
                uint64_l.append(tup)
            else:
                float64_l.append(tup)

        return tuple(int64_l + uint64_l + float64_l)


# it would be simpler just to wrap a structured array, but I read something
# online suggesting that structured arrays may have problems being sent over
# mpi4py using the interface for numpy arrays
#
# A hard and fast requirement is that numpy arrays are only allowed to have the
# alignment of np.float64, np.int64, and np.uint64

class ArrayMap(Mapping):
    """
    Represents a mapping of numpy arrays of fixed size.
    """

    def __init__(self, entry_spec, buffer = None):
        # users shouldn't specify buffer directly

        self._entry_spec = ArrayMapEntrySpec(entry_spec)

        buffer_shape = (self._entry_spec.required_storage_num_uint64(),)
        if buffer is None:
            # we use uint64 to get the alignment right
            self._buffer = np.empty(shape = buffer_shape, dtype = np.uint64)
        else:
            assert buffer.dtype == np.dtype(np.uint64)
            assert buffer.flags['C_CONTIGUOUS']
            assert buffer.shape == buffer_shape
            self._buffer = buffer

    @property
    def entry_spec(self):
        return self._entry_spec

    @property
    def data_buffer(self):
        return self._buffer

    def get_int64_buffer(self):
        out = self._buffer[self._entry_spec.get_dtype_slice(np.int64)]
        out.dtype = np.int64
        return out

    def get_float64_buffer(self):
        out = self._buffer[self._entry_spec.get_dtype_slice(np.float64)]
        out.dtype = np.float64
        return out

    def __getitem__(self, key):
        return self._entry_spec._get_subarray(key, self._buffer)

    def __len__(self):
        return len(self._entry_spec)

    def __iter__(self):
        for e in self._entry_spec:
            yield e[0]

    def asdict(self):
        return dict(self.items())

    @classmethod
    def copy_from_dict(cls, kv):
        entry_spec = []
        for k,v in kv.items():
            entry_spec.append((k, v.dtype, v.shape))
        out = cls(entry_spec)
        for k,v in kv.items():
            out[k][...] = v
        return out

    def update_from_other(self, other):
        # updates values held by self with values copied from other.
        #
        # This is generally an inefficient way to handle things...

        raise RuntimeError("Untested")
        if not isinstance(other, ArrayMap):
            other = ArrayMap.copy_from_dict(other)
        assert self.entry_spec == other.entry_spec
        self.data_buffer[:] = other.data_buffer[:]

    def _show_contents(self):
        """Method for debugging purposes"""
        l = [f"{self.__class__.__name__}{'{'}"]
        for key,val in self.items():
            l.append(f"  {key!r} : {val},")
        l.append("}")
        print(*l, sep = '\n')
