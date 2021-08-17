import numpy as np
from scipy.spatial.distance import pdist, cdist
import functools

import pyvsf

# implement a python-version of vsf_props that just uses numpy and scipy

_vsf_scalar_python_dict = {
    'mean' : [('mean', np.mean)],
    'variance' : [('mean', np.mean),
                  ('variance', functools.partial(np.var, ddof =1))],
}

def _vsf_props_python(pos_a, pos_b, vel_a, vel_b, dist_bin_edges,
                      statistic = 'variance', kwargs = {}):

    if pos_b is None and vel_b is None:
        distances = pdist(pos_a.T, 'euclidean')
        vdiffs = pdist(vel_a.T, 'euclidean')
    else:
        distances = cdist(pos_a.T, pos_b.T, 'euclidean')
        vdiffs = cdist(vel_a.T, vel_b.T, 'euclidean')

    num_bins = dist_bin_edges.size - 1
    bin_indices = np.digitize(x = distances,bins = dist_bin_edges)

    val_dict = {}
    if statistic == 'histogram':
        val_bin_edges = np.asanyarray(kwargs['val_bin_edges'],
                                      dtype = np.float64)
        val_dict['2D_counts'] = np.empty((num_bins,val_bin_edges.size - 1),
                                         dtype = np.int64)
        def _process_spatial_bin(spatial_bin_index, selected_vdiffs):
            hist, bin_edges = np.histogram(selected_vdiffs,
                                           bins = val_bin_edges)
            val_dict['2D_counts'][i,:] = hist
    else:
        stat_pair_l = _vsf_scalar_python_dict[statistic]
        val_dict = {}
        for (quantity_name, func) in stat_pair_l:
            val_dict[quantity_name] = np.empty((num_bins,), dtype = np.float64)
        val_dict['counts'] = np.empty((num_bins,), dtype = np.int64)

        def _process_spatial_bin(spatial_bin_index, selected_vdiffs):
            for (quantity_name, func) in stat_pair_l:
                if selected_vdiffs.size == 0:
                    val = np.nan
                else:
                    val = func(selected_vdiffs)
                val_dict[quantity_name][i] = val
            val_dict['counts'][spatial_bin_index] = selected_vdiffs.size

    for i in range(num_bins):
        # we need to add 1 to the i when checking for bin indices because
        # np.digitize assigns indices of 0 to values that fall to the left of
        # the first bin
        w = (bin_indices == (i+1))
        _process_spatial_bin(spatial_bin_index = i,
                             selected_vdiffs = vdiffs[w])
    return val_dict


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
                                 dist_bin_edges, kwargs = {},
                                 atol = 0.0, rtol = 0.0):

    alt_val_dict = _vsf_props_python(
        pos_a = pos_a, pos_b = pos_b, vel_a = vel_a, vel_b = vel_b,
        dist_bin_edges = dist_bin_edges, statistic = statistic,
        kwargs = kwargs
    )

    actual_val_dict = pyvsf.vsf_props(
        pos_a = pos_a, pos_b = pos_b, vel_a = vel_a, vel_b = vel_b,
        dist_bin_edges = dist_bin_edges, statistic = statistic,
        kwargs = kwargs
    )

    # check that the val_dicts have the same keys
    actual_key_set = frozenset(actual_val_dict.keys())
    if len(actual_key_set.symmetric_difference(alt_val_dict.keys())) != 0:
        raise AssertionError(
            "The python version's dict output has the keys, "
            f"{list(alt_val_dict.keys())}. In contrast, pyvsf.vsf_prop's "
            f"output dict has the keys, {list(actual_val_dict.keys())}"
        )

    _matching_array_dtype = lambda arr, dtype : isinstance(arr.flat[0], dtype)

    float_key_set = set()
    for key in actual_key_set:
        if _matching_array_dtype(actual_val_dict[key], np.floating):
            float_key_set.add(key)

    atol_dict = _prep_tol_dict(float_key_set, atol, "atol")
    rtol_dict = _prep_tol_dict(float_key_set, rtol, "rtol")

    for key in actual_val_dict.keys():

        if alt_val_dict[key].dtype != actual_val_dict[key].dtype:
            raise AssertionError(
                f'The actual "{key}" entries have dtype, {actual_dtype}, but '
                f'the reference entries have dtype, {alt_dtype}'
            )

        if _matching_array_dtype(actual_val_dict[key], np.integer):
            np.testing.assert_equal(
                actual = actual_val_dict[key], desired = alt_val_dict[key],
                err_msg = (f'The "{key}" entries of the output_dict are not '
                           'the same')
            )
        elif _matching_array_dtype(actual_val_dict[key], np.floating):
            np.testing.assert_allclose(
                actual = actual_val_dict[key],
                desired = alt_val_dict[key],
                equal_nan = True,
                rtol = rtol_dict[key],
                atol = atol_dict[key],
                err_msg = (f'The "{key}" entries of the output_dict are not '
                           'equal to within the specified tolerance'),
                verbose = True
            )
        else:
            raise NotImplementedError(
                "Unclear how to compare the contents of arrays with dtype = "
                f"{actual_vals.dtype}"
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

    val_bin_edges = np.array([-1.7976931348623157e+308, 1e-8,
                              1e-4, 1.7976931348623157e+308])

    if True: # simple case!
        x_a, vel_a = (np.arange(6.0).reshape(3,2),
                      np.arange(-3.0,3.0).reshape(3,2))

        x_b,vel_b = (np.arange(6.0,24.0).reshape(3,6),
                     np.arange(-9.0,9.0).reshape(3,6))

        bin_edges = np.array([17.0, 21.0, 25.0])

        _stat_quadruple = [
            ('variance', {}, 0.0, {'mean' : 2e-16, 'variance' : 1e-15}),
            ('histogram', {"val_bin_edges" : val_bin_edges}, 0.0, 0.0)
        ]

        for statistic, kwargs, atol, rtol in _stat_quadruple:
            _compare_vsf_implementations(pos_a = x_a, pos_b = x_b,
                                         vel_a = vel_a, vel_b = vel_b,
                                         statistic = statistic,
                                         dist_bin_edges = bin_edges,
                                         kwargs = kwargs,
                                         atol = atol,
                                         rtol = rtol)

    if True: # complex case:
        MY_SEED  = 156
        generator = np.random.RandomState(seed = MY_SEED)

        x_a, vel_a = _generate_vals((3,1000), generator)
        x_b, vel_b = _generate_vals((3,2000), generator)
        bin_edges = np.arange(11.0)/10
        _stat_quadruple = [
            ('variance', {}, 0.0, {'mean' : 2e-14, 'variance' : 3e-14}),
            ('histogram', {"val_bin_edges" : val_bin_edges}, 0.0, 0.0)
        ]

        for statistic, kwargs, atol, rtol in _stat_quadruple:
            _compare_vsf_implementations(pos_a = x_a, pos_b = x_b,
                                         vel_a = vel_a, vel_b = vel_b,
                                         statistic = statistic,
                                         dist_bin_edges = bin_edges,
                                         kwargs = kwargs,
                                         atol = atol,
                                         rtol = rtol)

def test_vsf_single_collection():

    val_bin_edges = np.array([-1.7976931348623157e+308, 1e-8,
                              1e-4, 1.7976931348623157e+308])

    if True: 
        x_a,vel_a = (np.arange(6.0,24.0).reshape(3,6),
                     np.arange(-9.0,9.0).reshape(3,6))
        bin_edges = np.array([0.0, 5.0, 10.0])

        _stat_quadruple = [
            ('variance', {}, 0.0, {'mean' : 0.0, 'variance' : 0.0}),
            ('histogram', {"val_bin_edges" : val_bin_edges}, 0.0, 0.0)
        ]

        for statistic, kwargs, atol, rtol in _stat_quadruple:
            _compare_vsf_implementations(pos_a = x_a, pos_b = None,
                                         vel_a = vel_a, vel_b = None,
                                         statistic = statistic,
                                         dist_bin_edges = bin_edges,
                                         kwargs = kwargs,
                                         atol = atol,
                                         rtol = rtol)
    if True: # complex case:
        MY_SEED  = 156
        generator = np.random.RandomState(seed = MY_SEED)

        x_a, vel_a = _generate_vals((3,1000), generator)
        bin_edges = np.arange(11.0)/10

        _stat_quadruple = [
            ('variance', {}, 0.0, {'mean' : 1e-14, 'variance' : 2e-14}),
            ('histogram', {"val_bin_edges" : val_bin_edges}, 0.0, 0.0)
        ]

        for statistic, kwargs, atol, rtol in _stat_quadruple:
        
            _compare_vsf_implementations(pos_a = x_a, pos_b = None,
                                         vel_a = vel_a, vel_b = None,
                                         statistic = statistic,
                                         dist_bin_edges = bin_edges,
                                         kwargs = kwargs,
                                         atol = atol,
                                         rtol = rtol)

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


def test_vsf_histograms():
    pass

if __name__ == '__main__':
    test_vsf_single_collection()
    test_vsf_two_collections()

    #benchmark((3,10000), shape_b = None, seed = 156,
    #          dist_bin_edges = np.arange(101.0)/100)

    #benchmark((3,20000), shape_b = None, seed = 156,
    #          dist_bin_edges = np.arange(101.0)/100)

    #benchmark((3,30000), shape_b = None, seed = 156,
    #          dist_bin_edges = np.arange(101.0)/100)

    #benchmark((3,50000), shape_b = None, seed = 156,
    #          dist_bin_edges = np.arange(101.0)/100)

    if False: # simple case!
        x_a, vel_a = (np.arange(6.0).reshape(3,2),
                      np.arange(-3.0,3.0).reshape(3,2))

        x_b,vel_b = (np.arange(6.0,24.0).reshape(3,6),
                     np.arange(-9.0,9.0).reshape(3,6))

        bin_edges = np.array([17.0, 21.0, 25.0])
        print(_vsf_props_python(x_a, x_b, vel_a, vel_b, bin_edges,
                                statistic = 'histogram'))
        print(pyvsf.vsf_props(x_a, x_b, vel_a, vel_b, bin_edges,
                              statistic = 'histogram'))
