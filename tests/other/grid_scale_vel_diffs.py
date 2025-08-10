from functools import partial

import numpy as np
import yt

from pairstat import vsf_props
from pairstat.small_dist_sf_props import grid_scale_vel_diffs
from pairstat._kernels import BulkAverage, BulkVariance

from bulk_statistics import compare_bulkstat, _kv_pair_cmp_iter, setup_ds


import functools


class _NeighborOpExecutor:
    def __init__(self, axis):
        self.x_slcs = slice(None), slice(None)
        self.y_slcs = slice(None), slice(None)
        self.z_slcs = slice(None), slice(None)

        if axis == "x":
            self.x_slcs = slice(0, -1, 1), slice(1, None, 1)
        elif axis == "y":
            self.y_slcs = slice(0, -1, 1), slice(1, None, 1)
        elif axis == "z":
            self.z_slcs = slice(0, -1, 1), slice(1, None, 1)
        else:
            raise ValueError(f"invalid axis value: {axis}")

    def __call__(self, op, field, grid):
        x_slc0, x_slc1 = self.x_slcs
        y_slc0, y_slc1 = self.y_slcs
        z_slc0, z_slc1 = self.z_slcs

        vals_0 = grid[field][x_slc0, y_slc0, z_slc0]
        vals_1 = grid[field][x_slc1, y_slc1, z_slc1]
        return op(vals_0, vals_1)


def _neighbor_vec_differences(grid, axis, diff_type, field_components):
    """
    Computes differences between vectors

    Parameters
    ----------
    diff_type : str
        Accepts values of 'parallel', 'transverse', 'total'

    Notes
    -----
    The different values accepted by diff_type haven the
    following meanings:
        - 'parallel': compute the signed difference of the
          vector component parallel to the displacement
          vector between 2 cells
        - 'transverse': compute the magnitude of the
          2 components perpendicular to the displacement
          vector between 2 cells
        - 'total': computes the magnitude of the differences
          between all components
    """
    exec_operation = _NeighborOpExecutor(axis)

    assert len(field_components) == 3

    if axis == "x":
        aligned_comp = field_components[0]
        transverse_comps = field_components[1:]
    elif axis == "y":
        aligned_comp = field_components[1]
        transverse_comps = field_components[2::-2]
    elif axis == "z":
        aligned_comp = field_components[2]
        transverse_comps = field_components[:2]
    else:
        raise ValueError(f"invalid axis value: {axis}")

    kernel = lambda x0, x1: x1 - x0
    if diff_type == "transverse":
        diff_l = [exec_operation(kernel, comp, grid) for comp in transverse_comps]
        return np.sqrt(np.square(diff_l[0]) + np.square(diff_l[1]))
    elif diff_type == "total":  # magnitude of velocity including all
        # 3 components
        diff_l = [exec_operation(kernel, comp, grid) for comp in field_components]
        return np.sqrt(
            np.square(diff_l[0]) + np.square(diff_l[1]) + np.square(diff_l[2])
        )
    elif diff_type == "parallel":  # signed difference of component parallel to
        # displacement vector between 2 cells
        return exec_operation(kernel, aligned_comp, grid)
    else:
        raise ValueError(f"invalid diff_type value: {diff_type}")


# when diff_type = 'parallel', negative values correspond to compression, and
# positive signs correspond to rarefaction
neighbor_vel_differences = functools.partial(
    _neighbor_vec_differences,
    field_components=[
        ("gas", "velocity_x"),
        ("gas", "velocity_y"),
        ("gas", "velocity_z"),
    ],
)


def _apply_neighbor_ops_to_field(grid, axis, field, op):
    exec_operation = _NeighborOpExecutor(axis)
    return exec_operation(op, field, grid)


max_neighboring_cs = functools.partial(
    _apply_neighbor_ops_to_field, field=("gas", "sound_speed"), op=np.maximum
)


def get_shared_logXe_bin_ind(grid, axis, logX_e_bin_edges):
    """
    Examines whether each pair of neighbors share a logXe bin.
    """

    def func(val0, val1):
        bins0 = np.digitize(val0.ndarray_view(), logX_e_bin_edges)
        out = np.empty_like(bins0)
        w = bins0 == np.digitize(val1.ndarray_view(), logX_e_bin_edges)
        out[w] = bins0[w]
        out[~w] = -1
        return out

    return _apply_neighbor_ops_to_field(grid, axis, ("gas", "logX_e"), func)


def _reference_calculation(ds, logX_e_bin_edges, hist_bin_edges):
    # this purely uses yt-machinery to complete the calculation

    name_pairs = [
        ("parallel", "aligned_vdiff"),
        ("transverse", "transverse_vdiff"),
        ("total", "mag_vdiff"),
    ]

    def cr_ind_iter():
        return range(0, len(phase_bin_edges) + 1)

    out = [{} for e in cr_ind_iter()]

    def update_results(min_mach_vel_diff, shared_logXe_bin_ind, name):
        for cr_ind in cr_ind_iter():
            counts = np.histogram(
                min_mach_vel_diff[cr_ind == shared_logXe_bin_ind],
                bins=hist_bin_edges[name + "_edges"],
            )[0]
            out_key = name + "_counts"

            if out_key not in out[cr_ind]:
                out[cr_ind][out_key] = counts
            else:
                out[cr_ind][out_key] += counts

    # first, collect the velocity differences
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    measurements = dict((diff_type, []) for diff_type, _ in name_pairs)
    shared_logXe_bin_ind_l = []
    for axis in "xyz":
        print(f"reference calc {axis}-axis setup")
        shared_logXe_bin_ind = get_shared_logXe_bin_ind(
            grid, axis=axis, logX_e_bin_edges=logX_e_bin_edges
        ).ravel()
        max_cs_vals = max_neighboring_cs(grid, axis=axis)

        print(f"reference calc {axis}-axis vdiff")
        for diff_type, name in name_pairs:
            vel_diffs = neighbor_vel_differences(grid, axis=axis, diff_type=diff_type)
            min_mach_vel_diff = (vel_diffs / max_cs_vals).to("dimensionless")

            update_results(
                min_mach_vel_diff.ndarray_view().ravel(), shared_logXe_bin_ind, name
            )
    grid.clear_data()
    return out


def func(ds, phase_bin_edges):
    print("main calculation")
    my_cut_regions = [f'obj["logX_e"].v < {float(phase_bin_edges[0]):.15e}']
    for i in range(len(phase_bin_edges) - 1):
        my_cut_regions.append(
            f'(obj["logX_e"].v >= {float(phase_bin_edges[i]):.15e}) & '
            + f'(obj["logX_e"].v < {float(phase_bin_edges[i + 1]):.15e})'
        )
    my_cut_regions.append(f'(obj["logX_e"].v > {float(phase_bin_edges[-1]):.15e})')

    tmp = grid_scale_vel_diffs(ds, cut_regions=my_cut_regions)

    actual_results = tmp[0]
    hist_bin_edges = tmp[2]

    reference_results = _reference_calculation(
        ds, logX_e_bin_edges=phase_bin_edges, hist_bin_edges=hist_bin_edges
    )

    assert len(actual_results) == len(reference_results)

    n_mismatches = 0
    for cr_ind, actual_counts_dict in enumerate(actual_results):
        ref_counts_dict = reference_results[cr_ind]
        assert len(actual_counts_dict) == len(ref_counts_dict) == 3
        for key in [
            "aligned_vdiff_counts",
            "transverse_vdiff_counts",
            "mag_vdiff_counts",
        ]:
            if not (ref_counts_dict[key] == actual_counts_dict[key]).all():
                print(
                    f"\nThe '{key}' values for cr_ind {cr_ind} don't match "
                    "up with reference values"
                )
                print("reference:")
                print(ref_counts_dict[key])
                print("\nactual:")
                print(actual_counts_dict[key])
                n_mismatches += 1
    assert n_mismatches == 0


if __name__ == "__main__":
    ds = setup_ds()
    phase_bin_edges = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0]) / 12.0

    func(ds, phase_bin_edges)

    if False:
        aligned_edges = np.array(
            [-np.inf] + np.linspace(-3, 3, num=121).tolist() + [np.inf]
        )
        transverse_edges = np.array(np.linspace(0, 3, num=121).tolist() + [np.inf])
        hist_edges = {
            "aligned_vdiff_edges": aligned_edges,
            "transverse_vdiff_edges": transverse_edges,
            "mag_vdiff_edges": transverse_edges.copy(),
        }

        _reference_calculation(
            ds, logX_e_bin_edges=phase_bin_edges, hist_bin_edges=hist_edges
        )
