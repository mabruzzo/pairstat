import numpy as np

"""
The basic idea here is to come up with a set of functions for each calculatable
type of Structure Function object that can be used to abstract over 
consolidation and such...
"""

from ._kernels_cy import Variance

def get_kernel(statistic):
    if statistic == 'histogram':
        return Histogram
    elif statistic == 'mean':
        return Mean
    elif statistic == 'variance':
        return Variance
    else:
        raise ValueError(f"Unknown Statistic: {statistic}")


class Mean:
    output_keys = ('counts', 'mean')
    commutative_consolidate = False

    @classmethod
    def consolidate_stats(cls, *rslts):
        raise NotImplementedError()

class Histogram:
    output_keys = ('2D_counts')
    commutative_consolidate = True

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
