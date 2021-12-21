from copy import deepcopy
import numpy as np

from libc.stdint cimport int64_t

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

class Variance:
    output_keys = ('counts', 'mean', 'variance')
    commutative_consolidate = False

    @classmethod
    def consolidate_stats(cls, *rslts):
        return _consolidate_variance(rslts)


