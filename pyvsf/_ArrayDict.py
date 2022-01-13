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

    def required_storage_num_uint64(self):
        """
        specify the amount of storage space needed to store data for each entry
        as a multiple of np.dtype(np.uint64).itemsize
        """
        # specifies the amount of storage space (as 
        return max(map(lambda e: e[-1], self._entries))

    def num_bytes(self):
        return np.dtype(np.uint64).itemsize * num_equivalent_uint64

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
            raise KeyError("key")
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

        out = []
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
            out.append((key, dtype, array_shape, arr_start,
                        arr_start + n_entries))

        return tuple(out)
                

# it would be simpler just to wrap a structured array, but I read something
# online suggesting that structured arrays may have problems being sent over
# mpi4py using the interface for numpy arrays
#
# A hard and fast requirement is that numpy arrays are only allowed to have the
# alignement of np.float64, np.int64, and np.uint64

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

    def __getitem__(self, key):
        return self._entry_spec._get_subarray(key, self._buffer)

    def __len__(self):
        return len(self._entry_spec)

    def __iter__(self):
        for e in self._entry_spec:
            yield e[0]

    def asdict(self):
        return dict(self.items())

class ArrayofArrayMaps:

    def __init__(self, shape, entry_spec):
        _array_shape_validation(shape)
        self._entry_spec = ArrayMapEntrySpec(entry_spec)

        single_array_map_buffer_shape = \
            (self._entry_spec.required_storage_num_uint64(),)

        complete_shape = shape + single_array_map_buffer_shape

        self._arr = np.empty(shape = complete_shape, dtype = np.uint64)

    @property
    def shape(self):
        return self._arr.shape[:-1]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def entry_spec(self):
        return self._entry_spec

    @property
    def data_buffer(self):
        return self._arr

    def __getitem__(self, idx):
        if self.ndim == 1:
            assert isinstance(idx, int)
        else:
            assert isinstance(idx, tuple)
            assert len(idx) == self.ndim
            assert all(isinstance(e,int) for e in int)

        sub_buffer = self._arr[idx]
        return ArrayMap(self.entry_spec, buffer = sub_buffer)


entry_spec = [('counts', np.int64, (15,)), ('mean', np.float64, (15,)),
              ('variance', np.int64, (15,))]
x = ArrayofArrayMaps((8,), entry_spec)

print(x.data_buffer.itemsize * x.data_buffer.size)

import pickle

print(x.shape)

old = np.empty(x.shape, dtype = object)
print(old.shape)
for i in range(x.size):
    old[i] = x[i].asdict()

print(len(pickle.dumps(x)))
#print(pickle.dumps(old))
print(len(pickle.dumps(old)))


