from itertools import product
import re

import numpy as np
import pytest

from pairstat._kernels_cy import (
    consolidate_partial_results,
    get_statconf,
    _test_evaluate_statconf,
)
from pairstat.utils import weighted_variance


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


def _prep_entries(statconf, vals, weights, add_empty_entries=True):
    entries = []
    for i, val in enumerate(vals):
        if add_empty_entries:
            entries.append({})
        if weights is None:
            cur_weights = None
        else:
            cur_weights = [weights[i]]

        entries.append(_test_evaluate_statconf(statconf, [val], cur_weights))
        if add_empty_entries:
            entries.append({})
    return entries


def calc_from_statconf_consolidation(
    statconf, vals, weights=None, add_empty_entries=True, pre_accumulate_idx_l=None
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
        When an empty list is specified (the default), we consolidate all of the values
        at once. When it contains slices, we separately compute the partial result for
        the elements in each slice (and any remaining points not in any slice), first,
        and then we consolidate those partial results.
    """
    dist_bin_edges = np.array([0.0, 10000])

    def _get_weights(idx):
        if weights is None:
            return None
        return weights[idx]

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
                args = _prep_entries(
                    statconf, vals[idx], _get_weights(idx), add_empty_entries
                )
            partial_eval.append(
                consolidate_partial_results(
                    statconf, args, dist_bin_edges=dist_bin_edges
                )
            )
        args = partial_eval
        if not visited.all():
            args += _prep_entries(
                statconf, vals[~visited], _get_weights(~visited), add_empty_entries
            )
        out = consolidate_partial_results(statconf, args, dist_bin_edges=dist_bin_edges)
    else:
        out = consolidate_partial_results(
            statconf,
            _prep_entries(statconf, vals, weights, add_empty_entries),
            dist_bin_edges=dist_bin_edges,
        )
    statconf.postprocess_rslt(out)
    return out


def direct_compute_stats(statconf, vals, weights=None):
    if statconf.name == "mean":
        return {
            "counts": np.array([len(vals)]),
            "mean": np.array([np.mean(vals)]),
        }
    elif statconf.name == "weightedmean":
        pair = np.average(vals, weights=weights, returned=True)
        return {"weight_sum": pair[1], "mean": pair[0]}
    elif statconf.name == "variance":
        return {
            "counts": np.array([len(vals)]),
            "mean": np.array([np.mean(vals)]),
            "variance": np.array([np.var(vals, ddof=1)]),
        }
    elif statconf.name == "cmoment3":
        mean = np.mean(vals)
        return {
            "counts": np.array([len(vals)]),
            "mean": np.array([mean]),
            "variance": np.array([np.var(vals, ddof=1)]),
            "cmoment3": np.mean((vals - mean) ** 3),
        }
    elif re.match(r"^(weighted)?omoment\d$", statconf.name):
        order = int(statconf.name[-1])
        if order == 0:
            raise RuntimeError("can't handle omoment0")
        if statconf.name.startswith("weighted"):
            pair = np.average(vals, weights=weights, returned=True)
            out = {"weight_sum": pair[1], "omoment1": pair[0]}
        else:
            out = {
                "weight_sum": np.array([len(vals)]),
                "omoment1": np.array([np.mean(vals)]),
            }
        for i in range(2, order + 1):
            out[f"omoment{i}"] = np.array(
                [np.average(np.power(vals, i), weights=weights)]
            )
        return out
    elif statconf.name == "weightedvariance":
        triple = weighted_variance(vals, weights=weights, returned=True)
        return {"weight_sum": triple[2], "mean": triple[1], "variance": triple[0]}
    elif statconf.name == "histogram":
        bin_edges = statconf._kwargs()["val_bin_edges"]
        return {"2D_counts": np.histogram(vals, bins=bin_edges)[0][np.newaxis]}
    elif statconf.name == "weightedhistogram":
        bin_edges = statconf._kwargs()["val_bin_edges"]
        return {
            "2D_weight_sums": np.histogram(
                vals, bins=bin_edges, density=False, weights=weights
            )[0][np.newaxis]
        }
    raise RuntimeError("Can't handle specified statconf")


def _test_consolidate(statconf, vals, weights=None, tol_spec=None):
    vals = np.array(vals)
    n_vals = vals.shape[0]

    pre_accumulate_idx_l_vals = [
        [],
        # effectively equivalent to previous
        [slice(0, n_vals)],
        # effectively equivalent to previous
        [slice(0, n_vals - 1)],
        # this tests edge case where all partial results are empty
        [slice(0, 0), slice(0, n_vals)],
        # tests scenario where the first partial result is zero
        [slice(0, 1), slice(1, n_vals)],
        # tests the scenario where both partial results include multiple counts
        [
            slice(0, n_vals // 2),
            slice(n_vals // 2, n_vals),
        ],
    ]

    ref_result = _test_evaluate_statconf(statconf, vals, weights)
    statconf.postprocess_rslt(ref_result)

    for pre_accumulate_idx_l in pre_accumulate_idx_l_vals:
        actual_result = calc_from_statconf_consolidation(
            statconf, vals, weights=weights, pre_accumulate_idx_l=pre_accumulate_idx_l
        )
        assert_all_close(ref_result, actual_result, tol_spec=tol_spec)


def simple_vals():
    return np.arange(6.0)


def random_vals():
    generator = np.random.RandomState(seed=2562642346)
    return generator.uniform(
        low=-1.0, high=np.nextafter(1.0, np.inf, dtype=np.float64), size=100
    )


statconfs = [
    get_statconf("mean", {}),
    get_statconf("weightedmean", {}),
    get_statconf("variance", {}),
    get_statconf("weightedvariance", {}),
    get_statconf("omoment2", {}),
    get_statconf("weightedomoment2", {}),
    get_statconf("omoment3", {}),
    get_statconf("weightedomoment3", {}),
    get_statconf("omoment4", {}),
    get_statconf("weightedomoment4", {}),
    get_statconf("histogram", {"val_bin_edges": np.linspace(-7, 7.0, num=101)}),
    get_statconf(
        "weightedhistogram",
        {"val_bin_edges": np.linspace(-7, 7.0, num=101)},
    ),
]

testdata = [
    pytest.param(statconf, vals_fn(), id=f"{statconf.name}-{vals_fn.__name__}")
    for statconf, vals_fn in product(statconfs, [simple_vals, random_vals])
]


@pytest.mark.parametrize("statconf,vals", testdata)
def test_against_pyimpl(statconf, vals, request):
    testid = request.node.callspec.id
    weights = None
    if statconf.requires_weights:
        weights = np.arange(len(vals))[::-1] + 3
    ref_result = direct_compute_stats(statconf, vals, weights)
    actual_result = _test_evaluate_statconf(statconf, vals, weights)
    statconf.postprocess_rslt(actual_result)
    tol_spec = {}
    if statconf.name in ["weightedmean", "weightedvariance"]:
        tol_spec = {("mean", "rtol"): 2e-16}

    elif "omoment" in statconf.name:
        _order = int(statconf.name[-1])
        if statconf.name.startswith("weighted") and testid.endswith("simple_vals"):
            _tol_triples = [("omoment1", "rtol", 2e-16)]
        elif testid.endswith("simple_vals"):
            _tol_triples = [("omoment4", "rtol", 2e-16)]
        elif statconf.name.startswith("weighted"):
            _tol_triples = [
                ("omoment1", "rtol", 2e-16),
                ("omoment3", "rtol", 3e-16),
                ("omoment4", "rtol", 2e-16),
            ]
        else:
            _tol_triples = [("omoment2", "rtol", 2e-16), ("omoment3", "rtol", 2e-16)]
        for dset, tolkind, val in _tol_triples:
            if int(dset[-1]) <= _order:
                tol_spec[dset, tolkind] = val

    assert_all_close(ref_result, actual_result, tol_spec=tol_spec)


@pytest.mark.parametrize("statconf,vals", testdata)
def test_consolidate(statconf, vals, request):
    testid = request.node.callspec.id

    weights = None
    if statconf.requires_weights:
        weights = np.arange(len(vals))[::-1] + 3

    tol_spec = {}
    if statconf.name == "variance" and testid.endswith("random_vals"):
        tol_spec = {("variance", "rtol"): 2e-16}

    elif statconf.name in ["weightedmean", "weightedvariance"]:
        if testid.endswith("simple_vals"):
            tol_spec = {("mean", "rtol"): 2e-16}
            if statconf.name == "weightedvariance":
                tol_spec["variance", "rtol"] = 4e-16
        else:
            tol_spec = {("mean", "rtol"): 6e-16}
            if statconf.name == "weightedvariance":
                tol_spec["variance", "rtol"] = 4e-16

    elif "omoment" in statconf.name:
        _order = int(statconf.name[-1])
        if statconf.name.startswith("weighted") and testid.endswith("simple_vals"):
            _tol_triples = [("omoment1", "rtol", 2e-16), ("omoment3", "rtol", 2e-16)]
        elif testid.endswith("simple_vals"):
            _tol_triples = [("omoment4", "rtol", 2e-16)]
        elif statconf.name.startswith("weighted"):
            _tol_triples = [
                ("omoment1", "rtol", 6e-16),
                ("omoment2", "rtol", 7e-16),
                ("omoment3", "rtol", 2e-15),
                ("omoment4", "rtol", 3e-16),
            ]
        else:
            _tol_triples = [
                ("omoment2", "rtol", 4e-16),
                ("omoment3", "rtol", 4e-16),
                ("omoment4", "rtol", 2e-16),
            ]

        for dset, tolkind, val in _tol_triples:
            if int(dset[-1]) <= _order:
                tol_spec[dset, tolkind] = val
    _test_consolidate(statconf, vals, weights, tol_spec=tol_spec)
