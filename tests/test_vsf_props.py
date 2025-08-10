from collections.abc import Sequence
from functools import partial
import time

import numpy as np
from scipy.spatial.distance import pdist, cdist
import pairstat


def zip_equal(*args):
    # more_itertools.zip_always started warning about deprecated and I like
    # that it eagerly evaluates so we implement our own version of it
    if len(args) > 1:
        nominal_length = len(args[0])
        for i, arg in enumerate(args):
            if len(arg) != nominal_length:
                raise ValueError(f"argument {i} doesn't have the same length as arg 0")
    return zip(*args)


# implement a python-version of vsf_props that just uses numpy and scipy

_vsf_scalar_python_dict = {
    "mean": [("mean", np.mean)],
    "variance": [("mean", np.mean), ("variance", partial(np.var, ddof=1))],
}


def _vsf_props_python(
    pos_a, pos_b, val_a, val_b, dist_bin_edges, stat_kw_pairs=[("variance", {})]
):
    if len(stat_kw_pairs) == 0:
        raise ValueError("At least one statistic must be specified")

    if pos_b is None and val_b is None:
        distances = pdist(pos_a.T, "euclidean")
        vdiffs = pdist(val_a.T, "euclidean")
    else:
        distances = cdist(pos_a.T, pos_b.T, "euclidean")
        vdiffs = cdist(val_a.T, val_b.T, "euclidean")

    num_bins = dist_bin_edges.size - 1
    bin_indices = np.digitize(x=distances, bins=dist_bin_edges)

    out = []
    for stat_name, stat_kw in stat_kw_pairs:
        out.append({})
        val_dict = out[-1]
        if stat_name == "histogram":
            val_bin_edges = np.asanyarray(stat_kw["val_bin_edges"], dtype=np.float64)
            val_dict["2D_counts"] = np.empty(
                (num_bins, val_bin_edges.size - 1), dtype=np.int64
            )

            def _process_spatial_bin(spatial_bin_index, selected_vdiffs):
                hist, bin_edges = np.histogram(selected_vdiffs, bins=val_bin_edges)
                val_dict["2D_counts"][i, :] = hist
        else:
            stat_pair_l = _vsf_scalar_python_dict[stat_name]
            for quantity_name, func in stat_pair_l:
                val_dict[quantity_name] = np.empty((num_bins,), dtype=np.float64)
            val_dict["counts"] = np.empty((num_bins,), dtype=np.int64)

            def _process_spatial_bin(spatial_bin_index, selected_vdiffs):
                for quantity_name, func in stat_pair_l:
                    if selected_vdiffs.size == 0:
                        val = np.nan
                    else:
                        val = func(selected_vdiffs)
                    val_dict[quantity_name][i] = val
                val_dict["counts"][spatial_bin_index] = selected_vdiffs.size

        for i in range(num_bins):
            # we need to add 1 to the i when checking for bin indices because
            # np.digitize assigns indices of 0 to values that fall to the left
            # of the first bin
            w = bin_indices == (i + 1)
            _process_spatial_bin(spatial_bin_index=i, selected_vdiffs=vdiffs[w])
    return out


def _call_separately_for_each_stat_pair(
    func, *args, stat_kw_pairs=[("variance", {})], **kwargs
):
    return [func(*args, stat_kw_pairs=[pair], **kwargs)[0] for pair in stat_kw_pairs]


VSF_PROPS_IMPL_REGISTRY = {
    "actual": (pairstat.vsf_props, "pairstat.vsf_props"),
    "actual-3proc-seq": (
        partial(pairstat.vsf_props, nproc=3, force_sequential=True),
        "pairstat.vsf_props(nproc=3, force_sequential = True)",
    ),
    "actual-3proc": (
        partial(pairstat.vsf_props, nproc=3, force_sequential=False),
        "pairstat.vsf_props(nproc=3, force_sequential = False)",
    ),
    "individual-stats": (
        partial(_call_separately_for_each_stat_pair, func=pairstat.vsf_props),
        "the individual-stats modified version",
    ),
    "python": (_vsf_props_python, "the python implementation"),
}


def _prep_tol_dict(key_set, tol_arg, tol_arg_name):
    if isinstance(tol_arg, dict):
        if len(key_set.symmetric_difference(tol_arg.keys())) != 0:
            raise ValueError(
                f'the dict passed to the "{tol_arg_name}" kwarg of '
                "_compare_vsf_implementations should only have the following "
                f"keys: {list(key_set)}. Instead, it has the keys: "
                f"{list(tol_arg.keys())}"
            )
        return tol_arg
    else:
        return dict((key, tol_arg) for key in key_set)


def _compare_vsf_implementation_single_rslt(
    alt_val_dict,
    actual_val_dict,
    atol=0.0,
    rtol=0.0,
    alt_impl_name=None,
    actual_impl_name=None,
):
    """
    Compares the results for a single statistic that is computed by 2 separate
    vsf implementations

    Parameters
    ----------
    alt_impl_name: str, optional
        Used to optionally provide context during test failures
    actual_impl_name: str, optional
        Used to optionally provide context during test failures
    """

    if alt_impl_name is None:
        alt_impl_name = "the alternate implementation"
    if actual_impl_name is None:
        actual_impl_name = "the actual implementation"

    # check that the val_dicts have the same keys
    actual_key_set = frozenset(actual_val_dict.keys())
    if len(actual_key_set.symmetric_difference(alt_val_dict.keys())) != 0:
        raise AssertionError(
            f"{alt_impl_name}'s output dict has the keys, "
            f"{list(alt_val_dict.keys())}. In contrast, {actual_impl_name}'s "
            f"output dict has the keys, {list(actual_val_dict.keys())}"
        )

    def _matching_array_dtype(arr, dtype):
        return isinstance(arr.flat[0], dtype)

    float_key_set = set()
    for key in actual_key_set:
        if _matching_array_dtype(actual_val_dict[key], np.floating):
            float_key_set.add(key)

    atol_dict = _prep_tol_dict(float_key_set, atol, "atol")
    rtol_dict = _prep_tol_dict(float_key_set, rtol, "rtol")

    for key in actual_val_dict.keys():
        actual_dtype = actual_val_dict[key].dtype
        alt_dtype = alt_val_dict[key].dtype
        if alt_dtype != actual_dtype:
            raise AssertionError(
                f'The {actual_impl_name}\'s "{key}" entry has the dtype, '
                f"{actual_dtype}, while the {alt_impl_name}'s entry has the "
                f"dtype, {alt_dtype}."
            )

        if _matching_array_dtype(actual_val_dict[key], np.integer):
            np.testing.assert_equal(
                actual=actual_val_dict[key],
                desired=alt_val_dict[key],
                err_msg=(f'The "{key}" entries of the output_dicts are not the same'),
            )
        elif _matching_array_dtype(actual_val_dict[key], np.floating):
            np.testing.assert_allclose(
                actual=actual_val_dict[key],
                desired=alt_val_dict[key],
                equal_nan=True,
                rtol=rtol_dict[key],
                atol=atol_dict[key],
                err_msg=(
                    f'The "{key}" entries of the output_dict are not '
                    "equal to within the specified tolerance"
                ),
                verbose=True,
            )
        else:
            raise NotImplementedError(
                "Unclear how to compare the contents of arrays with dtype = "
                f"{actual_dtype}"
            )


def compare_vsf_implementations(
    pos_a,
    pos_b,
    val_a,
    val_b,
    dist_bin_edges,
    stat_kw_pairs,
    atol=0.0,
    rtol=0.0,
    alt_implementation_key="python",
):
    # TODO: improve specification of separate atol and rtol values for
    # different stat_kw_pairs. The current approach is kind of dumb
    # - you need to provide a list with an element for each pair in
    #   stat_kw_pairs. If a 'scalar value' is provided that will apply to all
    #   statistics
    # - A 'scalar value' can be a number or a dict that associates tolerance
    #   values with the names of different quantities associated with the
    #   statistic

    assert alt_implementation_key != "actual"
    alt_impl_func, alt_impl_name = VSF_PROPS_IMPL_REGISTRY[alt_implementation_key]

    alt_rslt_l = alt_impl_func(
        pos_a=pos_a,
        pos_b=pos_b,
        val_a=val_a,
        val_b=val_b,
        dist_bin_edges=dist_bin_edges,
        stat_kw_pairs=stat_kw_pairs,
    )

    actual_rslt_l = pairstat.vsf_props(
        pos_a=pos_a,
        pos_b=pos_b,
        val_a=val_a,
        val_b=val_b,
        dist_bin_edges=dist_bin_edges,
        stat_kw_pairs=stat_kw_pairs,
    )

    def get_cur_tol(tol, index):
        if isinstance(tol, Sequence) and not isinstance(tol, dict):
            assert len(tol) == len(alt_rslt_l) == len(actual_rslt_l)
            return tol[index]
        return tol

    iter_tup = zip_equal(stat_kw_pairs, alt_rslt_l, actual_rslt_l)
    for i, ((stat_name, _), alt_rslt, actual_rslt) in enumerate(iter_tup):
        _compare_vsf_implementation_single_rslt(
            alt_rslt,
            actual_rslt,
            atol=get_cur_tol(atol, i),
            rtol=get_cur_tol(rtol, i),
            alt_impl_name=alt_impl_name,
            actual_impl_name="pairstat.vsf_props",
        )


def _generate_simple_vals(shape):
    if shape == (3, 2):
        return
    elif shape == (3, 6):
        return
    else:
        raise ValueError()


def _generate_vals(shape, generator):
    pos = generator.rand(*shape)
    vel = generator.rand(*shape) * 2 - 1.0
    return pos, vel


def test_vsf_two_collections():
    print("running tests against python implementation")
    val_bin_edges = np.array(
        [-1.7976931348623157e308, 1e-8, 1e-4, 1.7976931348623157e308]
    )

    if True:  # simple case!
        x_a, val_a = (np.arange(6.0).reshape(3, 2), np.arange(-3.0, 3.0).reshape(3, 2))

        x_b, val_b = (
            np.arange(6.0, 24.0).reshape(3, 6),
            np.arange(-9.0, 9.0).reshape(3, 6),
        )

        bin_edges = np.array([17.0, 21.0, 25.0])

        _stat_quadruple = [
            ("variance", {}, 0.0, {"mean": 2e-16, "variance": 1e-15}),
            ("histogram", {"val_bin_edges": val_bin_edges}, 0.0, 0.0),
        ]

        # check the calculation of individual statistics
        for statistic, kwargs, atol, rtol in _stat_quadruple:
            compare_vsf_implementations(
                pos_a=x_a,
                pos_b=x_b,
                val_a=val_a,
                val_b=val_b,
                dist_bin_edges=bin_edges,
                stat_kw_pairs=[(statistic, kwargs)],
                atol=atol,
                rtol=rtol,
            )

        # now, check the calculation of all statistics at once
        stat_l, kw_l, atol_l, rtol_l = zip(*_stat_quadruple)
        stat_kw_pairs = list(zip(stat_l, kw_l))
        compare_vsf_implementations(
            pos_a=x_a,
            pos_b=x_b,
            val_a=val_a,
            val_b=val_b,
            dist_bin_edges=bin_edges,
            stat_kw_pairs=stat_kw_pairs,
            atol=atol_l,
            rtol=rtol_l,
        )

    if True:  # complex case:
        MY_SEED = 156
        generator = np.random.RandomState(seed=MY_SEED)

        x_a, val_a = _generate_vals((3, 1000), generator)
        x_b, val_b = _generate_vals((3, 2000), generator)
        bin_edges = np.arange(11.0) / 10
        _stat_quadruple = [
            ("variance", {}, 0.0, {"mean": 2e-14, "variance": 3e-14}),
            ("histogram", {"val_bin_edges": val_bin_edges}, 0.0, 0.0),
        ]

        # check the calculation of individual statistics
        for statistic, kwargs, atol, rtol in _stat_quadruple:
            compare_vsf_implementations(
                pos_a=x_a,
                pos_b=x_b,
                val_a=val_a,
                val_b=val_b,
                dist_bin_edges=bin_edges,
                stat_kw_pairs=[(statistic, kwargs)],
                atol=atol,
                rtol=rtol,
            )

        # now, check the calculation of all statistics at once
        stat_l, kw_l, atol_l, rtol_l = zip(*_stat_quadruple)
        stat_kw_pairs = list(zip(stat_l, kw_l))
        compare_vsf_implementations(
            pos_a=x_a,
            pos_b=x_b,
            val_a=val_a,
            val_b=val_b,
            dist_bin_edges=bin_edges,
            stat_kw_pairs=stat_kw_pairs,
            atol=atol_l,
            rtol=rtol_l,
        )


def test_vsf_single_collection():
    print("running tests against python implementation")
    val_bin_edges = np.array(
        [0] + np.geomspace(start=1e-16, stop=100, num=100).tolist()
    )

    if True:
        x_a, val_a = (
            np.arange(6.0, 24.0).reshape(3, 6),
            np.arange(-9.0, 9.0).reshape(3, 6),
        )
        bin_edges = np.array([0.0, 5.0, 10.0])

        _stat_quadruple = [
            ("variance", {}, 0.0, {"mean": 0.0, "variance": 0.0}),
            ("histogram", {"val_bin_edges": val_bin_edges}, 0.0, 0.0),
        ]

        # check the calculation of individual statistics
        for statistic, kwargs, atol, rtol in _stat_quadruple:
            compare_vsf_implementations(
                pos_a=x_a,
                pos_b=None,
                val_a=val_a,
                val_b=None,
                dist_bin_edges=bin_edges,
                stat_kw_pairs=[(statistic, kwargs)],
                atol=atol,
                rtol=rtol,
            )

        # now, check the calculation of all statistics at once
        stat_l, kw_l, atol_l, rtol_l = zip(*_stat_quadruple)
        stat_kw_pairs = list(zip(stat_l, kw_l))
        compare_vsf_implementations(
            pos_a=x_a,
            pos_b=None,
            val_a=val_a,
            val_b=None,
            dist_bin_edges=bin_edges,
            stat_kw_pairs=stat_kw_pairs,
            atol=atol_l,
            rtol=rtol_l,
        )

    if True:  # complex case:
        MY_SEED = 156
        generator = np.random.RandomState(seed=MY_SEED)

        x_a, val_a = _generate_vals((3, 1000), generator)
        bin_edges = np.arange(11.0) / 10

        _stat_quadruple = [
            ("variance", {}, 0.0, {"mean": 1e-14, "variance": 2e-14}),
            ("histogram", {"val_bin_edges": val_bin_edges}, 0.0, 0.0),
        ]

        # check the calculation of individual statistics
        for statistic, kwargs, atol, rtol in _stat_quadruple:
            compare_vsf_implementations(
                pos_a=x_a,
                pos_b=None,
                val_a=val_a,
                val_b=None,
                dist_bin_edges=bin_edges,
                stat_kw_pairs=[(statistic, kwargs)],
                atol=atol,
                rtol=rtol,
            )

        # now, check the calculation of all statistics at once
        stat_l, kw_l, atol_l, rtol_l = zip(*_stat_quadruple)
        stat_kw_pairs = list(zip(stat_l, kw_l))
        compare_vsf_implementations(
            pos_a=x_a,
            pos_b=None,
            val_a=val_a,
            val_b=None,
            dist_bin_edges=bin_edges,
            stat_kw_pairs=stat_kw_pairs,
            atol=atol_l,
            rtol=rtol_l,
        )


def extra_multiple_stats_test(
    alt_implementation_key="individual-stats",
    skip_variance=False,
    skip_auto_sf=False,
    use_tol=False,
):
    # a few extra tests to verify that there is no difference in the result
    # (this is some very simplistic, crude fuzzing)

    val_bin_edges = np.array(
        [0] + np.geomspace(start=1e-16, stop=100, num=100).tolist()
    )
    stat_kw_pairs = [("variance", {}), ("histogram", {"val_bin_edges": val_bin_edges})]
    if skip_variance:
        stat_kw_pairs = stat_kw_pairs[1:]

    for seed in [4162, 2354, 7468, 3563, 88567]:
        generator = np.random.RandomState(seed=seed)

        x_a, val_a = _generate_vals((3, 1000), generator)
        x_b, val_b = _generate_vals((3, 2000), generator)
        bin_edges = np.arange(11.0) / 10
        _stat_quadruple = [
            ("variance", {}, 0.0, {"mean": 3.5e-14, "variance": 3e-14}),
            ("histogram", {"val_bin_edges": val_bin_edges}, 0.0, 0.0),
        ]

        if use_tol:
            _, _, atol, rtol = zip(*_stat_quadruple)
        else:
            atol, rtol = 0.0, 0.0

        # auto-structure-function
        if not skip_auto_sf:
            # use x_b and val_b to make sure that we use multiple processes
            # when nproc>1
            compare_vsf_implementations(
                pos_a=x_b,
                pos_b=None,
                val_a=val_b,
                val_b=None,
                dist_bin_edges=bin_edges,
                stat_kw_pairs=stat_kw_pairs,
                atol=atol,
                rtol=rtol,
                alt_implementation_key=alt_implementation_key,
            )
        return None
        # cross-structure-function
        compare_vsf_implementations(
            pos_a=x_a,
            pos_b=x_b,
            val_a=val_a,
            val_b=val_b,
            dist_bin_edges=bin_edges,
            stat_kw_pairs=stat_kw_pairs,
            atol=atol,
            rtol=rtol,
            alt_implementation_key=alt_implementation_key,
        )


def test_bundle_stats():
    print("running extra tests to check for problems with bundling stats")
    extra_multiple_stats_test()


def test_smp():
    print("checking partitioning for shared-memory multiprocessing")
    extra_multiple_stats_test(
        alt_implementation_key="actual-3proc-seq",
        skip_variance=False,
        skip_auto_sf=False,
        use_tol=True,
    )
    extra_multiple_stats_test(
        alt_implementation_key="actual-3proc",
        skip_variance=False,
        skip_auto_sf=False,
        use_tol=True,
    )


def benchmark(
    shape_a, shape_b=None, seed=156, skip_python_version=False, nproc=1, **kwargs
):
    generator = np.random.RandomState(seed=seed)

    pos_a, val_a = _generate_vals(shape_a, generator)
    if shape_b is None:
        pos_b, val_b = None, None
    else:
        pos_b, val_b = _generate_vals(shape_b, generator)

    # first, benchmark pairstat.vsf_props
    pairstat.vsf_props(
        pos_a=pos_a, pos_b=pos_b, val_a=val_a, val_b=val_b, nproc=nproc, **kwargs
    )
    t0 = time.perf_counter()
    pairstat.vsf_props(
        pos_a=pos_a, pos_b=pos_b, val_a=val_a, val_b=val_b, nproc=nproc, **kwargs
    )
    t1 = time.perf_counter()
    dt = t1 - t0
    print(f"pairstat.vsf_props version: {dt} seconds")

    if not skip_python_version:
        # second, benchmark scipy/numpy version:
        _vsf_props_python(pos_a=pos_a, pos_b=pos_b, val_a=val_a, val_b=val_b, **kwargs)
        t0 = time.perf_counter()
        _vsf_props_python(pos_a=pos_a, pos_b=pos_b, val_a=val_a, val_b=val_b, **kwargs)
        t1 = time.perf_counter()
        dt = t1 - t0
        print(f"Scipy/Numpy version: {dt} seconds")


def test_benchmark(capsys):
    with capsys.disabled():
        print(
            "\nrunning a short auto-vsf benchmark. This takes ~7 s on a 4 core system"
        )
        val_bin_edges = np.geomspace(start=1e-16, stop=2.0, num=100, dtype=np.float64)
        val_bin_edges[0] = 0.0
        val_bin_edges[-1] = np.finfo(np.float64).max
        benchmark(
            (3, 20000),
            shape_b=None,
            seed=156,
            dist_bin_edges=np.arange(101.0) / 100,
            stat_kw_pairs=[
                ("histogram", {"val_bin_edges": val_bin_edges}),
                ("variance", {}),
            ],
            nproc=4,
            skip_python_version=True,
        )

        print(
            "\nrunning a short cross-vsf benchmark. This takes ~10 s on a 4 core system"
        )
        benchmark(
            (3, 20000),
            shape_b=(3, 20000),
            seed=156,
            dist_bin_edges=np.arange(101.0) / 100,
            stat_kw_pairs=[
                ("histogram", {"val_bin_edges": val_bin_edges}),
                ("variance", {}),
            ],
            nproc=4,
            skip_python_version=True,
        )
