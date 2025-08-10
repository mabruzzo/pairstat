# this only exists to ease testing

from libc.stdint cimport uint64_t
from libc.stddef cimport size_t

cdef extern from "partition.hpp":
    ctypedef struct StatTask:
        uint64_t start_A
        uint64_t stop_A
        uint64_t start_B
        uint64_t stop_B

    cdef cppclass TaskIt:
        bint has_next()
        StatTask next()

    cdef cppclass TaskItFactory:
        TaskItFactory(size_t nproc, size_t n_points, size_t n_points_other,
                      bint skip_small_prob_check)
        uint64_t n_partitions()
        TaskIt* build_TaskIt_ptr(size_t proc_id)


cdef class _PyStatTask:
    cdef StatTask val

    @staticmethod
    cdef create(StatTask val):
        out = _PyStatTask()
        out.val = val
        return out

    @property
    def start_A(self): return self.val.start_A

    @property
    def stop_A(self): return self.val.stop_A

    @property
    def start_B(self): return self.val.start_B

    @property
    def stop_B(self): return self.val.stop_B

    def __str__(self):
        return ('{' + f'"start_A":{self.start_A}, "stop_A":{self.stop_A}, '
                + f'"start_B":{self.start_B}, "stop_B":{self.stop_B}"' + '}')

cdef class _PyTaskIt:
    cdef TaskIt* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if (self.ptr != NULL):
            del self.ptr

    def has_next(self):
        assert self.ptr != NULL
        return self.ptr.has_next()

    def __next__(self):
        if not self.has_next():
            raise StopIteration()
        cdef StatTask tmp = self.ptr.next()
        return _PyStatTask.create(tmp)

    @staticmethod
    cdef create(TaskIt* ptr):
        out = _PyTaskIt()
        out.ptr = ptr
        return out

    def __iter__(self):
        return self

cdef class _PyTaskItFactory:
    cdef TaskItFactory* ptr

    def __cinit__(self, size_t nproc, size_t n_points,
                  size_t n_points_other = 0,
                  bint skip_small_prob_check = False):
        self.ptr = new TaskItFactory(nproc, n_points, n_points_other,
                                     skip_small_prob_check)

    def __dealloc__(self):
        del self.ptr

    def num_partitions(self):
        return self.ptr.n_partitions()

    def build_iterator(self, proc_id):
        cdef TaskIt* tmp = self.ptr.build_TaskIt_ptr(proc_id)
        return _PyTaskIt.create(tmp)

def build_task_it_factory(nproc, n_points, n_points_other = 0,
                          skip_small_prob_check = False):
    return _PyTaskItFactory(nproc, n_points, n_points_other,
                            skip_small_prob_check)
