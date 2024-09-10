import numpy as np
import pytest

from pyvsf._kernels_cy import (
    get_statconf,
    consolidate_partial_results,
    _test_evaluate_statconf,
)


def assert_all_close(ref, actual, tol_spec=None):
    __tracebackhide__ = True
    if tol_spec is None:
        tol_spec = {}
    if len(ref) != len(actual):
        pytest.fail("ref and actual don't have the same keys")

    accessed_tol_count = 0

    def _access_tol(key, tol_kind):
        nonlocal accessed_tol_count
        try:
            out = tol_spec[key, tol_kind]
            accessed_tol_count += 1
            return out
        except KeyError:
            return 0.0

    for key in ref:
        if key not in actual:
            pytest.fail(f"actual is missing {key}")
        rtol = _access_tol(key, "rtol")
        atol = _access_tol(key, "atol")
        if (rtol == 0) and (atol == 0):
            np.testing.assert_array_equal(
                ref[key], actual[key], err_msg=f"the {key!r} vals aren't equal"
            )
        else:
            np.testing.assert_allclose(
                actual[key],
                ref[key],
                rtol=rtol,
                atol=atol,
                err_msg=f"the {key!r} vals aren't equal",
            )
    if accessed_tol_count != len(tol_spec):
        raise RuntimeError("something went very wrong with the specified tolerances!")


def _prep_entries(statconf, vals, add_empty_entries=True):
    l = []
    for val in vals:
        if add_empty_entries:
            l.append({})
        l.append(_test_evaluate_statconf(statconf, [val]))
        if add_empty_entries:
            l.append({})
    return l


def calc_from_statconf_consolidation(
    statconf, vals, add_empty_entries=True, pre_accumulate_idx_l=None
):
    """
    Performs calculations using consolidation

    Essentially, we create an individual partial-result for every single value and then
    we use the statconf to consolidate them together

    Parameters
    ----------
    statconf
        Specifies the statistic being computed. For these calculations, we just use a
        single separation bin
    vals : array_like
        The sequence of values for which we are computing statistics
    add_empty_entries : bool
        Indicates whether we inject empty partial results
    pre_accumulate_idx_l : list of slice objects, optional
        When an empty list is specifed (the default), we consolidate all of the values
        at once. When it contains slices, we separately compute the partial result for
        the elements in each slice (and any remaining points not in any slice), first,
        and then we consolidate those partial results.
    """
    dist_bin_edges = np.array([0.0, 10000])
    if len(pre_accumulate_idx_l) != 0:
        vals = np.array(vals)
        num_vals = vals.shape[0]

        partial_eval = []
        visited = np.zeros((num_vals,), dtype=np.bool_)
        for i, idx in enumerate(pre_accumulate_idx_l):
            if vals[idx].size == 0:
                args = [{}, {}]
            elif visited[idx].any():
                raise RuntimeError(f"an index of {idx} has already been visited")
            else:
                visited[idx] = True
                args = _prep_entries(statconf, vals[idx], add_empty_entries)
            partial_eval.append(
                consolidate_partial_results(
                    statconf, args, dist_bin_edges=dist_bin_edges
                )
            )
        args = partial_eval
        if not visited.all():
            args += _prep_entries(statconf, vals[~visited], add_empty_entries)
        out = consolidate_partial_results(statconf, args, dist_bin_edges=dist_bin_edges)
    else:
        out = consolidate_partial_results(
            statconf,
            _prep_entries(statconf, vals, add_empty_entries),
            dist_bin_edges=dist_bin_edges,
        )
    statconf.postprocess_rslt(out)
    return out


def direct_compute_stats(statconf, vals):
    if statconf.name == "mean":
        return {
            "counts": np.array([len(vals)]),
            "mean": np.array([np.mean(vals)]),
        }
    elif statconf.name == "variance":
        return {
            "counts": np.array([len(vals)]),
            "mean": np.array([np.mean(vals)]),
            "variance": np.array([np.var(vals, ddof=1)]),
        }
    raise RuntimeError("Can't handle specified statconf")


def _test_consolidate(statconf, vals, tol_spec=None):
    vals = np.array(vals)
    n_vals = vals.shape[0]

    pre_accumulate_idx_l_vals = [
        [],
        [slice(0, n_vals)],  # effectively equivalent to previous
        [slice(0, n_vals - 1)],  # effectively equivalent to previous
        [slice(0, 0), slice(0, n_vals)],  # this tests edge case where all
        # partial results are empty
        [slice(0, 1), slice(1, n_vals)],  # tests scenario where the first partial
        # result is zero
        [
            slice(0, n_vals // 2),  # tests the scenario where both partial
            slice(n_vals // 2, n_vals),
        ],  # results include multiple counts
    ]

    ref_result = direct_compute_stats(statconf, vals)

    for pre_accumulate_idx_l in pre_accumulate_idx_l_vals:
        actual_result = calc_from_statconf_consolidation(
            statconf, vals, pre_accumulate_idx_l=pre_accumulate_idx_l
        )
        assert_all_close(ref_result, actual_result, tol_spec=tol_spec)


@pytest.fixture
def simple_vals():
    return np.arange(6.0)


@pytest.fixture
def random_vals():
    generator = np.random.RandomState(seed=2562642346)
    return generator.uniform(
        low=-1.0, high=np.nextafter(1.0, np.inf, dtype=np.float64), size=100
    )


def test_consolidate_variance_simple(simple_vals):
    statconf = get_statconf("variance", {})
    _test_consolidate(statconf, simple_vals)


def test_consolidate_variance_random(random_vals):
    statconf = get_statconf("variance", {})
    _test_consolidate(statconf, random_vals, {("variance", "rtol"): 2e-16})
