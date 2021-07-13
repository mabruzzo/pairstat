import numpy as np
from scipy.spatial.distance import pdist, cdist
import functools

import pyvsf

# implement a python-version of vsf_props that just uses numpy and scipy

_vsf_python_dict = {
    'mean' : [('mean', np.mean)],
    'variance' : [('mean', np.mean),
                  ('variance', functools.partial(np.var, ddof =1))],
}

def _vsf_props_python(pos_a, pos_b, vel_a, vel_b, dist_bin_edges,
                      statistic = 'variance'):

    if pos_b is None and vel_b is None:
        distances = pdist(pos_a.T, 'euclidean')
        vdiffs = pdist(vel_a.T, 'euclidean')
    else:
        distances = cdist(pos_a.T, pos_b.T, 'euclidean')
        vdiffs = cdist(vel_a.T, vel_b.T, 'euclidean')

    num_bins = dist_bin_edges.size - 1
    bin_indices = np.digitize(x = distances,bins = dist_bin_edges)

    counts = np.empty((num_bins,), dtype = np.int64)

    stat_pair_l = _vsf_python_dict[statistic]
    val_dict = {}
    for (quantity_name, func) in stat_pair_l:
        val_dict[quantity_name] = np.empty((num_bins,), dtype = np.float64)

    means = []
    variances = []

    for i in range(num_bins):
        # we need to add 1 to the i when checking for bin indices because
        # np.digitize assigns indices of 0 to values that fall to the left of
        # the first bin
        w = (bin_indices == (i+1))
        counts[i] = w.sum()
        for (quantity_name, func) in stat_pair_l:
            if counts[i] == 0:
                val = np.nan
            else:
                
                val = func(vdiffs[w])
            val_dict[quantity_name][i] = val
    return counts, val_dict


def _prep_tol_dict(key_set, tol_arg, tol_arg_name):
    if isinstance(tol_arg,dict):
        if len(key_set.symmetric_difference(tol_arg.keys())) != 0:
            raise ValueError(
                f'the dict passed to the "{tol_arg_name}" kwarg of '
                '_compare_vsf_implementations should only have the following '
                f'keys: {list(key_set)}. Instead, it has the keys: '
                f'{list(tol_arg.keys())}'
            )
        return tol_arg
    else:
        return dict((key,tol_arg) for key in key_set)

def _compare_vsf_implementations(pos_a, pos_b, vel_a, vel_b, statistic,
                                 dist_bin_edges,
                                 atol = 0.0, rtol = 0.0):
    alt_counts, alt_val_dict = _vsf_props_python(
        pos_a = pos_a, pos_b = pos_b, vel_a = vel_a, vel_b = vel_b,
        dist_bin_edges = dist_bin_edges, statistic = statistic
    )

    actual_counts, actual_val_dict = pyvsf.vsf_props(
        pos_a = pos_a, pos_b = pos_b, vel_a = vel_a, vel_b = vel_b,
        dist_bin_edges = dist_bin_edges, statistic = statistic
    )

    # check that the same number of entries have been added to each bin
    np.testing.assert_equal(
        actual = actual_counts, desired = alt_counts,
        err_msg = ("Both versions of the code must find the same number of "
                   "entries per bin"),
        verbose = True
    )

    # check that the val_dicts have the same keys
    actual_key_set = frozenset(actual_val_dict.keys())
    if len(actual_key_set.symmetric_difference(alt_val_dict.keys())) != 0:
        raise AssertionError(
            "The python version's dict output has the keys, "
            f"{list(alt_val_dict.keys())}. In contrast, pyvsf.vsf_prop's "
            f"output dict has the keys, {list(actual_val_dict.keys())}"
        )

    atol_dict = _prep_tol_dict(actual_key_set, atol, "atol")
    rtol_dict = _prep_tol_dict(actual_key_set, rtol, "rtol")

    for key in actual_val_dict.keys():
        np.testing.assert_allclose(
            actual = actual_val_dict[key],
            desired = alt_val_dict[key],
            equal_nan = True,
            rtol = rtol_dict[key],
            atol = atol_dict[key],
            err_msg = (f'The "{key}" entries of the output_dict are not equal '
                       'to within the tolerance'),
            verbose = True
        )


def _generate_simple_vals(shape):
    if shape == (3,2):
        return 
    elif shape == (3,6):
        return 
    else:
        raise ValueError()

def _generate_vals(shape, generator):
    pos = generator.rand(*shape)
    vel = generator.rand(*shape)*2 - 1.0
    return pos,vel

# now define the actual tests!
    
def test_vsf_two_collections():

    if True: # simple case!
        x_a, vel_a = (np.arange(6.0).reshape(3,2),
                      np.arange(-3.0,3.0).reshape(3,2))

        x_b,vel_b = (np.arange(6.0,24.0).reshape(3,6),
                     np.arange(-9.0,9.0).reshape(3,6))

        bin_edges = np.array([17.0, 21.0, 25.0])


        _compare_vsf_implementations(pos_a = x_a, pos_b = x_b,
                                     vel_a = vel_a, vel_b = vel_b,
                                     statistic = 'variance',
                                     dist_bin_edges = bin_edges,
                                     atol = 0.0,
                                     rtol = {'mean' : 2e-16,
                                             'variance' : 1e-15})

    if True: # complex case:
        MY_SEED  = 156
        generator = np.random.RandomState(seed = MY_SEED)

        x_a, vel_a = _generate_vals((3,1000), generator)
        x_b, vel_b = _generate_vals((3,2000), generator)
        bin_edges = np.arange(11.0)/10
        _compare_vsf_implementations(pos_a = x_a, pos_b = x_b,
                                     vel_a = vel_a, vel_b = vel_b,
                                     statistic = 'variance',
                                     dist_bin_edges = bin_edges,
                                     atol = 0.0,
                                     rtol = {'mean' : 2e-14,
                                             'variance' : 3e-14})

def test_vsf_single_collection():

    if True: 
        x_a,vel_a = (np.arange(6.0,24.0).reshape(3,6),
                     np.arange(-9.0,9.0).reshape(3,6))
        bin_edges = np.array([0.0, 5.0, 10.0])
        _compare_vsf_implementations(pos_a = x_a, pos_b = None,
                                     vel_a = vel_a, vel_b = None,
                                     statistic = 'variance',
                                     dist_bin_edges = bin_edges,
                                     atol = 0.0,
                                     rtol = {'mean' : 0.0, 'variance' : 0.0})
    if True: # complex case:
        MY_SEED  = 156
        generator = np.random.RandomState(seed = MY_SEED)

        x_a, vel_a = _generate_vals((3,1000), generator)
        bin_edges = np.arange(11.0)/10
        _compare_vsf_implementations(pos_a = x_a, pos_b = None,
                                     vel_a = vel_a, vel_b = None,
                                     statistic = 'variance',
                                     dist_bin_edges = bin_edges,
                                     atol = 0.0,
                                     rtol = {'mean' : 1e-14,
                                             'variance' : 2e-14})

import time
def benchmark(shape_a, shape_b = None, seed = 156, **kwargs):
    generator = np.random.RandomState(seed = seed)

    pos_a, vel_a = _generate_vals(shape_a, generator)
    if shape_b is None:
        pos_b, vel_b = None, None
    else:
        pos_b, vel_b = _generate_vals(shape_b, generator)

    # first, benchmark pyvsf.vsf_props
    pyvsf.vsf_props(pos_a = pos_a, pos_b = pos_b,
                      vel_a = vel_a, vel_b = vel_b,
                      **kwargs)
    t0 = time.perf_counter()
    pyvsf.vsf_props(pos_a = pos_a, pos_b = pos_b,
                    vel_a = vel_a, vel_b = vel_b,
                    **kwargs)
    t1 = time.perf_counter()
    dt = t1 - t0
    print(f"pyvsf.vsf_props version: {dt} seconds")

    # second, benchmark scipy/numpy version:
    _vsf_props_python(pos_a = pos_a, pos_b = pos_b,
                      vel_a = vel_a, vel_b = vel_b,
                      **kwargs)
    t0 = time.perf_counter()
    _vsf_props_python(pos_a = pos_a, pos_b = pos_b,
                      vel_a = vel_a, vel_b = vel_b,
                      **kwargs)
    t1 = time.perf_counter()
    dt = t1 - t0
    print(f"Scipy/Numpy version: {dt} seconds")

if __name__ == '__main__':
    test_vsf_single_collection()
    test_vsf_two_collections()

    benchmark((3,10000), shape_b = None, seed = 156,
              dist_bin_edges = np.arange(101.0)/100)

    benchmark((3,20000), shape_b = None, seed = 156,
              dist_bin_edges = np.arange(101.0)/100)

    benchmark((3,30000), shape_b = None, seed = 156,
              dist_bin_edges = np.arange(101.0)/100)

    benchmark((3,50000), shape_b = None, seed = 156,
              dist_bin_edges = np.arange(101.0)/100)
