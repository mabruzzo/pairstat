"""
Define some statistical kernels that are unrelated to the calculation of the
structure function
"""

from copy import deepcopy

import numpy as np

from .utils import weighted_variance
from ._kernels_cy import _validate_basic_quan_props, _allocate_unintialized_rslt_dict


def is_field_type(arg):
    try:
        n_elem = len(arg)
    except TypeError:
        return False
    if n_elem != 2:
        return False
    return all(isinstance(e, str) for e in arg)


def _extract_weight_unit_pairs(kwargs, skip_checks=False):
    """
    Reads the kwargs dict and extract a list of pairs of were the first entry
    of each pair is the name of a weight_field and the second entry is the
    unit for that field
    """

    if not skip_checks:
        if len(kwargs) != 1:
            raise ValueError("kwargs should be a dictionary holding 1 element")
        assert "weight_field" in kwargs

        pair_l = kwargs["weight_field"]
        n_pairs = len(pair_l)
        if n_pairs == 0:
            raise ValueError("kwargs['weight_field'] must hold 1+ entries")
        elif (n_pairs == 2) and is_field_type(pair_l[0]) and isinstance(pair_l[1], str):
            raise ValueError(
                "kwargs['weight_field'] should be set to a list of pairs of "
                "(field_name, field_units). It seems that the user may have "
                "forgotten to enclose a single pair in a list"
            )

        for pair in pair_l:
            assert len(pair) == 2
            assert is_field_type(pair[0])
            assert isinstance(pair[1], str)

    return kwargs["weight_field"]


def _generic_kernel_handle_args(
    quan, extra_quantities, kwargs, list_of_weight_fields=False
):
    """
    Helper function that factors out some features for use in the main
    functions used to compute non-structure function statistics.

    This returns a list of the appropriate weight_fields (take from
    extra_quantities) in the expected order
    """
    if quan.size == 0:
        return {}
    elif np.ndim(quan) != 2:
        raise ValueError("quan must be a 2D array")
    assert quan.shape[0] == 3

    out = []
    pairs = _extract_weight_unit_pairs(kwargs, skip_checks=True)
    for weight_field_name, _ in pairs:
        assert weight_field_name in extra_quantities
        weights = extra_quantities[weight_field_name]
        assert np.ndim(weights) == 1
        assert quan.shape[1] == weights.shape[0]
        out.append(weights)
    return out


def compute_bulkaverage(quan, extra_quantities, kwargs):
    """
    Parameters
    ----------
    quan: np.ndarray
        Expected to be a (3,N) array of doubles that nominally hold N velocity
        values. This is not a unyt array.
    extra_quan: dict
        Dictionary where keys correspond to field names and values correspond
        to 1D arrays holding N values of that array (N should match
        quan.shape[1]).
    kwargs: dict
        This should be a 1-element dict. The key should be 'weight_field' and
        the value should be a tuple, where the first element specifies the name
        of the weight field and the second element specifies the expected units.

    """
    weight_l = _generic_kernel_handle_args(quan, extra_quantities, kwargs)
    assert len(weight_l) == 1
    weights = weight_l[0]

    # axis = 1 seems counter-intuitive, but I've confirmed it's correct
    averages, sum_of_weights = np.average(quan, axis=1, weights=weights, returned=True)

    assert averages.ndim == sum_of_weights.ndim == 1  # sanity check!

    # since sum_of_weights is 1D and has a length equal to quan.shape[1], all 3
    # entries are identical
    assert (sum_of_weights[0] == sum_of_weights).all()
    assert sum_of_weights[0] != 0.0  # we may want to revisit return vals if
    # untrue

    weight_total = np.array([[sum_of_weights[0]]], dtype=sum_of_weights.dtype)
    averages.shape = (1, averages.shape[0])

    return {"average": averages, "weight_total": weight_total}


class BulkAverage:
    """
    This is used to directly compute weight average values for velocity
    components.

    TODO: consider letting the number of components change
    TODO: consider handling multiple weight fields at once
    TODO: consider handling no weight field
    """

    name = "bulkaverage"
    operate_on_pairs = False
    non_vsf_func = compute_bulkaverage
    # the following isn't a required class attribute, it's just a common choice
    output_keys = ("weight_total", "average")

    @classmethod
    def n_ghost_ax_end(cls):
        return 0

    @classmethod
    def get_extra_fields(cls, kwargs={}):
        weight_unit_pairs = _extract_weight_unit_pairs(kwargs)
        if len(weight_unit_pairs) != 1:
            raise ValueError("Currently only 1 weight field is supported.")

        weight_field_name, weight_field_units = weight_unit_pairs[0]
        return {weight_field_name: (weight_field_units, cls.operate_on_pairs)}

    @classmethod
    def get_dset_props(cls, dist_bin_edges, kwargs={}):
        weight_unit_pairs = _extract_weight_unit_pairs(kwargs)
        n_weight_fields = len(weight_unit_pairs)
        if n_weight_fields != 1:
            raise ValueError("Currently only 1 weight field is supported.")
        return [
            (
                "weight_total",
                np.float64,
                (
                    n_weight_fields,
                    1,
                ),
            ),
            (
                "average",
                np.float64,
                (
                    n_weight_fields,
                    3,
                ),
            ),
        ]

    @classmethod
    def consolidate_stats(cls, *rslts):
        # we could run into overflow problems.
        # We're using a compensated summation
        accum_prodsum = np.zeros((3,), np.float64)
        c_prodsum = np.zeros_like(accum_prodsum)  # needs to be zeros

        accum_wsum = np.zeros((1,), np.float64)
        c_wsum = np.zeros_like(accum_wsum)  # needs to be zeros

        first_filled_rslt = None
        for rslt in rslts:
            if len(rslt) == 0:
                continue
            if first_filled_rslt is None:
                first_filled_rslt = rslt

            assert len(rslt) == 2  # check the keys

            cur_weight = rslt["weight_total"][0, :]
            cur_product = cur_weight * rslt["average"][0, :]

            # first, accumulate weight
            cur_elem = cur_weight - c_wsum
            tmp_accum = accum_wsum + cur_elem
            c_wsum = (tmp_accum - accum_wsum) - cur_elem
            accum_wsum = tmp_accum

            # next, accumulate product
            cur_elem = cur_product - c_prodsum
            tmp_accum = accum_prodsum + cur_elem
            c_prodsum = (tmp_accum - accum_prodsum) - cur_elem
            accum_prodsum = tmp_accum

        if first_filled_rslt is None:
            return {}
        elif (accum_wsum[0] == first_filled_rslt["weight_total"][0, 0]).all():
            return {
                "weight_total": first_filled_rslt["weight_total"].copy(),
                "average": first_filled_rslt["average"].copy(),
            }
        else:
            weight_total, weight_times_avg = accum_wsum, accum_prodsum
            weight_total.shape = (1, 1)
            weight_times_avg.shape = (1, 3)
            return {
                "weight_total": weight_total,
                "average": weight_times_avg / weight_total,
            }

    @classmethod
    def validate_rslt(cls, rslt, dist_bin_edges, kwargs={}):
        _validate_basic_quan_props(cls, rslt, dist_bin_edges, kwargs)

    @classmethod
    def postprocess_rslt(cls, rslt, kwargs={}):
        pass  # do nothing

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, kwargs={}, postprocess_rslt=True):
        rslt = _allocate_unintialized_rslt_dict(cls, dist_bin_edges, kwargs)
        for k in rslt.keys():
            if k == "average":
                rslt[k][:] = np.nan
            else:
                rslt[k][:] = 0.0
        if postprocess_rslt:
            cls.postprocess_rslt(rslt, kwargs=kwargs)
        return rslt


def compute_bulk_variance(quan, extra_quantities, kwargs):
    """
    Parameters
    ----------
    quan: np.ndarray
        Expected to be a (3,N) array of doubles that nominally hold N velocity
        values. This is not a unyt array.
    extra_quan: dict
        Dictionary where keys correspond to field names and values correspond
        to 1D arrays holding N values of that array (N should match
        quan.shape[1]).
    kwargs: dict
        This should be a 1-element dict. The key should be 'weight_field' and
        the value should be a tuple, where the first element specifies the name
        of the weight field and the second element specifies the expected units.

    """
    weight_l = _generic_kernel_handle_args(quan, extra_quantities, kwargs)
    n_weight_fields = len(weight_l)

    variance = np.empty((n_weight_fields, quan.shape[0]), dtype=np.float64)
    averages = np.empty_like(variance)
    weight_total = np.empty((n_weight_fields, 1), dtype=np.float64)

    for i, weights in enumerate(weight_l):
        # axis = 1 seems counter-intuitive, but I've confirmed it's correct
        cur_vars, cur_avgs, cur_sum_of_weights = weighted_variance(
            quan, axis=1, weights=weights, returned=True
        )

        # since weights is 1D and has a length equal to quan.shape[1], all 3
        # entries of cur_sum_of_weights are identical
        assert (cur_sum_of_weights[0] == cur_sum_of_weights).all()

        assert cur_sum_of_weights[0] != 0.0  # we may want to revisit return
        # vals if untrue

        weight_total[i, 0] = cur_sum_of_weights[0]
        averages[i, :] = cur_avgs
        variance[i, :] = cur_vars

    return {"variance": variance, "average": averages, "weight_total": weight_total}


class BulkVariance:
    """
    This is used to directly compute weight variance values for velocity
    components.

    This uses the variance equation without the correction for bias. Reasons
    are given below for why application of Bessel's correction doesn't make
    much sense.

    Notes
    -----
    I've given this extensive thought, and I've decided it doesn't make any
    sense to try to apply a variant of Bessel's correction to this calculation
    (to try and get an unbiased estimate of the variance).

    There are three main points here that are worth consideration:
    - Bessel's correction for an unweighted variance has a smaller and smaller
      effect as the number of samples increase (the result gets closer and
      closer to biased estimator). In other words, it's correcting for the bias
      that arises from undersampling.

      - as an aside, Bessel's correction will give you a different result if
        you artificially inflated the number of samples. For example,
        if you duplicated every sample, the estimate of the mean and biased
        estimate of the variance remains unchanged. However, the unbiased
        variance estimate (with Bessel's correction) gives a different result.

    - The weight field doesn't need to be related to the number of samples in
      any way. It just so happens that samples in a unigrid simulation may have
      a fixed volume and samples in a particle-based simulation may have fixed
      mass. These are 2 special cases where one could get an unbiased
      volume-weighted and unbiased mass-weighted variance, respectively.

      - Therefore, in the general case, Bessel's correction does NOT get
        applied to the weight field

    Here are 2 compelling cases to consider:
      1. Suppose we wanted to measure a mass-weighted velocity average from a
         unigrid simulation. Consider these 2 alternative scenarios:
           a) We have 3 cells with masses of 4 g, 10g, and 100 g

           b) We have 7 cells with masses of 1 g, a cell with a mass of 8 g,
              and a cell with a mass of 97 g.

         For the sake of argument, let's assume mass is independent of velocity.
         If that's the case, then it's obvious that a smaller bias-correction
         is needed for the second case even though the first case has more
         mass (and one might argue that there's more 'stuff' in the first case)

           - it also becomes clear that there is no way to apply Bessel's
             correction based on the mass, since it's the choice of units would
             alter the correction.

       2. Consider a 2D AMR simulation, whose domain is subdivided into the
          following 5 blocks:

              +----+----+---------+
              | 1a | 1b |         |
              +----+----+    2    |
              | 1c | 1d |         |
              +----+----+---------+

         Suppose that each refined blocks has 0.25 the area of the coarse block
         and covers a quarter of the area. An area-weighted variance for some
         arbitrary quantity, using all of the cells in any one of these blocks
         would have the same level of bias-correction even though block 2
         covers a larger area.
         - there's probably *some* clever analytic formula that could be
           applied for correcting bias in area-weighted variances over the full
           domain. But, again that would be a special case. The same could
           not be said for weighting by some other quantity like mass
    """

    name = "bulkvariance"
    operate_on_pairs = False
    non_vsf_func = compute_bulk_variance
    # the following isn't a required class attribute, it's just a common choice
    output_keys = ("weight_total", "average", "variance")

    @classmethod
    def n_ghost_ax_end(cls):
        return 0

    @classmethod
    def get_extra_fields(cls, kwargs={}):
        weight_unit_pairs = _extract_weight_unit_pairs(kwargs)

        out = {}
        for weight_field_name, weight_field_units in weight_unit_pairs:
            out[weight_field_name] = (weight_field_units, cls.operate_on_pairs)
        return out

    @classmethod
    def get_dset_props(cls, dist_bin_edges, kwargs={}):
        weight_unit_pairs = _extract_weight_unit_pairs(kwargs)
        n_weight_fields = len(weight_unit_pairs)

        return [
            ("weight_total", np.float64, (n_weight_fields, 1)),
            ("average", np.float64, (n_weight_fields, 3)),
            ("variance", np.float64, (n_weight_fields, 3)),
        ]

    @classmethod
    def consolidate_stats(cls, *rslts):
        # find first non-empty result
        sample = next(filter(lambda e: e != {}, rslts), None)
        if sample is None:
            return {}
        else:
            # initialize output dict:
            out = dict((k, np.zeros_like(v)) for k, v in sample.items())

        # concatenate entries from each dict into 3 larger arrays
        n_rslts = len(rslts)
        if n_rslts == 1:
            return deepcopy(rslts[0])
        all_weight_totals = np.empty(
            (n_rslts,) + sample["weight_total"].shape, dtype=np.float64
        )
        all_means = np.empty((n_rslts,) + sample["average"].shape, dtype=np.float64)
        all_variances = np.empty(
            (n_rslts,) + sample["variance"].shape, dtype=np.float64
        )

        assert all_weight_totals.shape[:-1] == all_means.shape[:-1]
        assert all_weight_totals.shape[-1] == 1
        assert all_means.shape == all_variances.shape

        for i, rslt in enumerate(rslts):
            if rslt == {}:
                all_weight_totals[i, ...] = 0.0
                # fill in the rest after the loop
            else:
                all_weight_totals[i, ...] = rslt["weight_total"]
                all_means[i, ...] = rslt["average"]
                all_variances[i, ...] = rslt["variance"]

        # for each location in all_weight_totals that is 0, make sure that
        # corresponding location in all_means and all_variances are also 0
        # (we want to make sure they're NOT NaNs)
        _mask = all_weight_totals == 0.0
        assert (_mask.shape[-1] == 1) and (_mask.ndim == all_means.ndim)
        _mask.shape = _mask.shape[:-1]
        all_means[_mask] = 0.0
        all_variances[_mask] = 0.0

        def func(local_weight_tots, local_means, local_vars):
            # this borrows heavily from the yt-project!
            dtype = np.float64

            global_weight_total = local_weight_tots.sum(dtype=dtype)
            if global_weight_total == 0.0:
                return np.nan, np.nan, 0.0

            global_mean = (local_weight_tots * local_means).sum(
                dtype=dtype
            ) / global_weight_total

            delta2 = np.square(local_means - global_mean, dtype=dtype)
            global_var = (local_weight_tots * (local_vars + delta2)).sum(
                dtype=dtype
            ) / global_weight_total

            return global_var, global_mean, global_weight_total

        assert all_variances.ndim == 3

        # outer loop indices correspond to calculations that used different
        # weight fields
        for weight_ind in range(all_variances.shape[-2]):
            # inner loop all used the same weight field
            for i in range(all_variances.shape[-1]):
                global_var, global_mean, global_weight_total = func(
                    all_weight_totals[:, weight_ind, 0],
                    all_means[:, weight_ind, i],
                    all_variances[:, weight_ind, i],
                )
                if i == 0:
                    out["weight_total"][weight_ind, 0] = global_weight_total
                else:
                    assert (
                        out["weight_total"][weight_ind, 0] == global_weight_total
                    )  # sanity check
                out["average"][weight_ind, i] = global_mean
                out["variance"][weight_ind, i] = global_var

        if (out["weight_total"] == 0.0).any():
            raise RuntimeError(
                "Encountered weight_total == 0. We may want to reconsider some things."
            )

        return out

    @classmethod
    def validate_rslt(cls, rslt, dist_bin_edges, kwargs={}):
        _validate_basic_quan_props(cls, rslt, dist_bin_edges, kwargs)

    @classmethod
    def postprocess_rslt(cls, rslt, kwargs={}):
        pass  # do nothing

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, kwargs={}, postprocess_rslt=True):
        rslt = _allocate_unintialized_rslt_dict(cls, dist_bin_edges, kwargs)
        for k in rslt.keys():
            if k in ["variance", "average"]:
                rslt[k][:] = np.nan
            else:
                rslt[k][:] = 0.0
        if postprocess_rslt:
            cls.postprocess_rslt(rslt, kwargs=kwargs)
        return rslt
