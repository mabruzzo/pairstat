import numpy as np

from pyvsf._kernels_cy import Variance as _Variance

def _prep_entries(vals, add_empty_entries = True):
    l = []
    for val in vals:
        if add_empty_entries:
            l.append({})
            l.append({'counts'   : np.array([1]),
                      'mean'     : np.array([val]),
                      'variance' : np.array([0.])})
            if add_empty_entries:
                l.append({})
    return l

def calc_variance_from_kernels(vals, add_empty_entries = True,
                               pre_accumulate_idx_l = []):
    if len(pre_accumulate_idx_l) != 0:
        vals = np.array(vals)
        num_vals = vals.shape[0]

        partial_eval = []
        visited = np.zeros((num_vals,), dtype = np.bool8)
        for i,idx in enumerate(pre_accumulate_idx_l):
            if vals[idx].size == 0:
                args = [{}, {}]
            elif visited[idx].any():
                raise RuntimeError(
                    f"an index of {idx} has already been visited"
                )
            else:
                visited[idx] = True
                args = _prep_entries(vals[idx], add_empty_entries)
            partial_eval.append(_Variance.consolidate_stats(*args))

        args = partial_eval
        if not visited.all():
            args += _prep_entries(vals[~visited], add_empty_entries)
        return _Variance.consolidate_stats(*args)
    else:
        return _Variance.consolidate_stats(
            *_prep_entries(vals, add_empty_entries)
        )

def direct_compute_stats(vals):
    return {'counts'   : np.array([len(vals)]),
            'mean'     : np.array([np.mean(vals)]),
            'variance' : np.array([np.var(vals, ddof = 1)])}

def test_consolidate_variance(vals, mean_rtol = 0.0, variance_rtol = 0.0):
    vals = np.array(vals)
    n_vals = vals.shape[0]
    
    pre_accumulate_idx_l_vals = [
        [],
        [slice(0,n_vals)],             # effectively equivalent to previous
        [slice(0,n_vals - 1)],         # effectively equivalent to previous
        [slice(0,0), slice(0,n_vals)], # this tests edge case where all
                                       # partial results are empty
        [slice(0,1), slice(1,n_vals)], # tests scenario where the first partial
                                       # result is zero
        [slice(0,n_vals//2),           # tests the scenario where both partial
         slice(n_vals//2,n_vals)],     # results include multiple counts
                                       
    ]

    ref_result = direct_compute_stats(vals)

    for pre_accumulate_idx_l in pre_accumulate_idx_l_vals:
        actual_result = calc_variance_from_kernels(
            vals, pre_accumulate_idx_l = pre_accumulate_idx_l
        )
        np.testing.assert_array_equal(ref_result['counts'],
                                      actual_result['counts'])
        np.testing.assert_allclose(ref_result['mean'],
                                   actual_result['mean'],
                                   atol = 0.0, rtol = mean_rtol)
        np.testing.assert_allclose(ref_result['variance'],
                                   actual_result['variance'],
                                   atol = 0.0, rtol = variance_rtol)


if __name__ == '__main__':
    vals = np.arange(6.)
    test_consolidate_variance(vals)

    generator = np.random.RandomState(seed = 2562642346)

    vals = generator.uniform(low = -1.0,
                             high = np.nextafter(1.0,np.inf,dtype = np.float64),
                             size = 100)
    test_consolidate_variance(vals, variance_rtol = 2e-16)


    # should generalize test so we can make vals into a 2D array
