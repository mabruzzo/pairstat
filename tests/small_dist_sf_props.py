import matplotlib.pyplot as plt
import numpy as np
import yt

from pyvsf import vsf_props
from pyvsf.small_dist_sf_props import BoxSelector, small_dist_sf_props


from gascloudtool.utilities import SetupDS

config_fname = 'X1000_M1.5_HD_CDdftFloor_R864.7_logP3.ini'
suffix = 'R8_X1000Floor'
step = 1/8.
_tmp_prefix = '/home/mabruzzo/Dropbox/research/turb_cloud/analysis/'
path = ('/media/mabruzzo/Elements/turb_cloud/cloud_runs/'
        'X1000_M1.5_HD_CDdftCstr_R864.7_logP3_Res8/cloud_07.5000/'
        'cloud_07.5000.block_list')
#path = ('/home/mabruzzo/research/turb_cloud/cloud_runs/'
#        'X1000_M1.5_HD_CDdftFloor_R864.7_logP3_Res8/cloud_07.5000/'
#        'cloud_07.5000.block_list')

setup_func = SetupDS(
    fname = f'{_tmp_prefix}/cloud_configs/{config_fname}',
    env_fname = f'{_tmp_prefix}/cloud_env'
)
ds = yt.load(path)
setup_func(ds)

bin_edges = np.array([0.2777777777777778, 0.5555555555555556,
                      0.8333333333333334, 1.1111111111111112,
                      1.3888888888888888])
my_cut_regions = [f'obj["logX_K"].v < {float(bin_edges[1]):.15e}',
                  #f'(obj["logX_K"].v >= {float(bin_edges[1]):.15e}) & ' +
                  #f'(obj["logX_K"].v < {float(bin_edges[3]):.15e})']
                  ]

component_fields = (('gas','velocity_x'),
                    ('gas','velocity_y'),
                    ('gas','velocity_z'))


def _cal_ref_vsf_props(dist_bin_edges, cut_regions, geometric_selector,
                       statistic, kwargs):
    print('computing the vsf_props directly:')
    if not isinstance(statistic, str):
        stat_kw_pairs = list(zip(statistic,kwargs))
        single_stat = False
    else:
        stat_kw_pairs = [(statistic, kwargs)]
        single_stat = True

    out = []
    for stat_name,_ in stat_kw_pairs:
        out.append([None for e in cut_regions])

    ad = geometric_selector.apply_selector(ds)
    for cr_ind, cut_region in enumerate(cut_regions):

        cad = ad.cut_region(cut_region)
        pos = np.array([cad[ii].to('cloud_radius').ndarray_view() \
                        for ii in ['x', 'y', 'z']])
        quan = np.array([cad[f].to('wind_velocity').ndarray_view() \
                         for f in component_fields])

        for stat_ind, (stat_name, kw) in enumerate(stat_kw_pairs):
            out[stat_ind][cr_ind] = vsf_props(pos_a = pos, vel_a = quan,
                                              pos_b = None, vel_b = None,
                                              dist_bin_edges = dist_bin_edges,
                                              statistic = stat_name,
                                              kwargs = kw)
    if single_stat:
        return out[0]
    return out

def compare(dist_bin_edges, cut_regions, geometric_selector, nproc = 1,
            statistic = 'histogram'):

    if not isinstance(statistic, str):
        statistic_l = statistic
    else:
        statistic_l = [statistic]

    kwargs_l = []
    for stat_name in statistic_l:
        if stat_name == 'histogram':
            vel_bin_edges = np.array(
                [0.0] + np.geomspace(1e-10, 1.0, num = 101).tolist() +
                [np.finfo(np.float64).max]
            )
            kwargs_l.append({'val_bin_edges' : vel_bin_edges})
        else:
            kwargs_l.append({})

    if len(statistic_l) == 1:
        statistic = statistic_l[0]
        kwargs = kwargs_l[0]
    else:
        statistic = statistic_l
        kwargs = kwargs_l
    
    ref = _cal_ref_vsf_props(dist_bin_edges, cut_regions, geometric_selector,
                             statistic, kwargs)

    print('computing the vsf_props from alternative approach')
    pool = None
    if nproc > 1:
        pool = schwimmbad.MultiPool(processes = nproc)
    
    tmp = small_dist_sf_props(
        ds, dist_bin_edges = dist_bin_edges,
        cut_regions = cut_regions,
        pos_units = "cloud_radius",
        quantity_units = "wind_velocity",
        component_fields = component_fields,
        max_points = None,
        rand_seed = None,
        geometric_selector = geometric_selector,
        statistic = statistic,
        kwargs = kwargs,
        pool = pool
    )

    other_rslt = tmp[0]
    points_used = tmp[1]
    subvol_decomp = tmp[3]
    return ref, other_rslt, points_used, subvol_decomp

def _kv_pair_cmp_iter(ref, actual):
    assert len(ref) == len(actual)
    assert all(k in actual for k in ref.keys())
    for k in ref.keys():
        yield k, ref[k], actual[k]

def compare_variance(ref, actual, mean_rtol = 0.0, variance_rtol = 0.0,
                     mean_atol = 0.0, variance_atol = 0.0):
    for key, r_vals, a_vals in _kv_pair_cmp_iter(ref, actual):
        if key == 'counts':
            np.testing.assert_array_equal(
                r_vals, a_vals,
                err_msg = "the 'counts' entries should be identical"
            )
            continue
        elif key == 'mean':
            rtol,atol = mean_rtol, mean_atol
        elif key == 'variance':
            rtol,atol = variance_rtol, variance_atol
        np.testing.assert_allclose(
            actual = a_vals,
            desired = r_vals,
            equal_nan = True,
            rtol = rtol,
            atol = atol,
            err_msg = (f'The "{key}" entries of the output_dict are not '
                       'equal to within the specified tolerance'),
            verbose = True
        )

def perform_comparison(statistic_l, compare_func, ref, actual):
    if isinstance(statistic_l, str):
        compare_func(statistic = statistic_l, ref = ref, actual = actual)
    else:
        for i,statistic in enumerate(statistic_l):
            assert isinstance(statistic, str)
            print("Comparing: " + statistic)
            compare_func(statistic = statistic, ref = ref[i],
                         actual = actual[i])

def test_single_subvol(statistic):
    # first, ensure that results are consistent when you use just 1 subvolume
    my_dist_bin_edges = np.arange(step*0.5, 3.5 + step, step)
    my_geometric_selector = BoxSelector(
        left_edge = [-2.0,-2.0,-2.0], right_edge = [2.0,2.0,2.0],
        length_unit = 'code_length',
    )
    ref, actual, points_used, subvol_decomp = compare(
        my_dist_bin_edges, my_cut_regions, my_geometric_selector,
        statistic = statistic
    )
    assert subvol_decomp.subvols_per_ax == (1,1,1)

    def compare_rslts(statistic, ref, actual):
        if statistic == 'histogram':
            assert (ref[0]['2D_counts'] == actual[0]['2D_counts']).all()
        elif statistic == 'variance':
            compare_variance(ref[0], actual[0],
                             mean_rtol = 0.0, variance_rtol = 0.0)
    perform_comparison(statistic, compare_rslts, ref = ref, actual = actual)

def test_two_subvol(statistic):
    for dim in range(3):
        # second let's compare the case where we have 2 subvolumes!
        my_dist_bin_edges = np.arange(step*0.5, 1.5 + step, step)
        if dim == 0:
            left_edge, right_edge = [-2.0,-1.0,-1.0], [2.0,1.0,1.0]
            subvols_per_ax = (2,1,1)
        elif dim == 1:
            left_edge, right_edge = [-1.0,-2.0,-1.0], [1.0,2.0,1.0]
            subvols_per_ax = (1,2,1)
        else:
            left_edge, right_edge = [-1.0,-1.0,-2.0], [1.0,1.0,2.0]
            subvols_per_ax = (1,1,2)
        my_geometric_selector = BoxSelector(
            left_edge = left_edge, right_edge = right_edge,
            length_unit = 'code_length',
        )
        ref, actual, points_used, subvol_decomp = compare(
            my_dist_bin_edges, my_cut_regions, my_geometric_selector,
            statistic = statistic
        )
        assert subvol_decomp.subvols_per_ax == subvols_per_ax
        def compare_rslts(statistic, ref, actual):
            if statistic == 'histogram':
                assert (ref[0]['2D_counts'] == actual[0]['2D_counts']).all()
            elif statistic == 'variance':
                compare_variance(ref[0], actual[0],
                                 mean_rtol = 0.0,
                                 variance_rtol = 0.0,
                                 mean_atol = [5.e-16, 3e-16, 3e-16][dim],
                                 variance_atol = [1e-17, 2e-17, 8e-18][dim])
        perform_comparison(statistic, compare_rslts, ref = ref, actual = actual)

def test_four_subvol(statistic):
    for i in range(3):
        # second let's compare the case where we have 2 subvolumes!
        my_dist_bin_edges = np.arange(step*0.5, 1.5 + step, step)
        if i == 0:
            left_edge, right_edge = [-2.0,-2.0,-1.0], [2.0,2.0,1.0]
            subvols_per_ax = (2,2,1)
        elif i == 1:
            left_edge, right_edge = [-2.0,-1.0,-2.0], [2.0,1.0,2.0]
            subvols_per_ax = (2,1,2)
        else:
            left_edge, right_edge = [-1.0,-2.0,-2.0], [1.0,2.0,2.0]
            subvols_per_ax = (1,2,2)
        my_geometric_selector = BoxSelector(
            left_edge = left_edge, right_edge = right_edge,
            length_unit = 'code_length',
        )
        ref, actual, points_used, subvol_decomp = compare(
            my_dist_bin_edges, my_cut_regions, my_geometric_selector,
            statistic = statistic
        )
        assert subvol_decomp.subvols_per_ax == subvols_per_ax
        def compare_rslts(statistic, ref, actual):
            if statistic == 'histogram':
                if not (ref[0]['2D_counts'] == actual[0]['2D_counts']).all():
                    raise ValueError(
                        "Histograms don't match.\n"
                        f"Expected {ref['2D_counts'].sum()} counts\n"
                        f"Found {actual[0]['2D_counts'].sum()} counts\n"
                    )
                elif statistic == 'variance':
                    compare_variance(ref[0], actual[0],
                                     mean_rtol = 0.0,
                                     variance_rtol = 0.0,
                                     mean_atol = [5.e-16, 5e-16, 5e-16][i],
                                     variance_atol = [9e-18, 9e-18, 2e-17][i]
                    )
        perform_comparison(statistic, compare_rslts, ref = ref, actual = actual)

def test_64_subvol(statistic):
    my_dist_bin_edges = np.arange(step*0.5, 1.5 + step, step)
    left_edge, right_edge = [-4.0,-4.0,-4.0], [4.0,4.0,4.0]
    subvols_per_ax = (4,4,4)
    my_geometric_selector = BoxSelector(
        left_edge = left_edge, right_edge = right_edge,
        length_unit = 'code_length',
    )

    cur_cut_regions = [f'obj["logX_K"].v < {float(bin_edges[1]):.15e}',
                       f'(obj["logX_K"].v >= {float(bin_edges[1]):.15e}) & ' +
                       f'(obj["logX_K"].v < {float(bin_edges[3]):.15e})']
    ref, actual, points_used, subvol_decomp = compare(
        my_dist_bin_edges, cur_cut_regions, my_geometric_selector,
        statistic = statistic
    )

    assert subvol_decomp.subvols_per_ax == subvols_per_ax
    def compare_rslts(statistic, ref, actual):
        if statistic == 'histogram':
            for i in range(len(cur_cut_regions)):
                if not (ref[i]['2D_counts'] == actual[i]['2D_counts']).all():
                    raise ValueError(
                        "Histograms don't match.\n"
                        f"Expected {ref[i]['2D_counts'].sum()} counts\n"
                        f"Found {actual[i]['2D_counts'].sum()} counts\n"
                    )
        elif statistic == 'variance':
            for cut_region_i in range(len(cur_cut_regions)):
                compare_variance(ref[cut_region_i], actual[cut_region_i],
                                 mean_rtol = 2.1e-14, variance_rtol = 2e-14,
                                 mean_atol = 0.0, variance_atol = 0.0)
    perform_comparison(statistic, compare_rslts, ref = ref, actual = actual)



if __name__ == '__main__':

    # perform some tests where we consider one statistic at a time
    for stat in ['histogram', 'variance']:
        test_single_subvol(stat)
        test_two_subvol(stat)
        test_four_subvol(stat)
        test_64_subvol(stat)

    # perform a test where we consider multiple statistics at the same time
    test_64_subvol(['histogram', 'variance'])
