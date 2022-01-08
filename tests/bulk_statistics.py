# test our implementation of the bulk statistic kernels against the
# derived_quantities machinery in yt

import numpy as np
import yt

from pyvsf.small_dist_sf_props import BoxSelector, small_dist_sf_props
from pyvsf._kernels import BulkAverage, BulkVariance

def setup_ds():

    from gascloudtool.utilities import SetupDS

    config_fname = 'X1000_M1.5_HD_CDdftFloor_R864.7_logP3.ini'

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
    return ds

def _kv_pair_cmp_iter(ref, actual):
    if len(ref) != len(actual):
        raise AssertionError(
            f"ref keys, {list(ref.keys())}, don't match actual keys, "
            f"{list(actual.keys())}"
        )
    assert all(k in actual for k in ref.keys())
    for k in ref.keys():
        yield k, ref[k], actual[k]

def compare_bulkstat(ref, actual,
                     weight_total_rtol = 0., weight_total_atol = 0.,
                     average_rtol = 0., average_atol = 0.,
                     variance_rtol = 0.0, variance_atol = 0.,
                     bulkvariance_cmp = False):
    for key, r_vals, a_vals in _kv_pair_cmp_iter(ref, actual):
        if key == 'weight_total':
            rtol,atol = weight_total_rtol, weight_total_atol
        elif key == 'average':
            rtol,atol = average_rtol, average_atol
        elif bulkvariance_cmp and key == 'variance':
            rtol,atol = variance_rtol, variance_atol
        else:
            raise RuntimeError(f"Unrecgonized output_dict key: {key}")
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

def _yt_calc_bulkstat(fields, ds, geometric_selector, cut_regions,
                      stat_kw_pairs, quantity_units = "cm/s"):

    out = [[None for e in cut_regions] for (_,_) in stat_kw_pairs]

    assert len(stat_kw_pairs) == 1
    stat_name, kw = stat_kw_pairs[0]
    assert stat_name in [BulkAverage.name, BulkVariance.name]

    ad = geometric_selector.apply_selector(ds)
    for cr_ind, cut_region in enumerate(cut_regions):
        cad = ad.cut_region(cut_region)
        assert cad['index','ones'].size != 0

        weight_field, weight_units = kw['weight_field']

        rslt = {}
        rslt['weight_total'] = np.array(
            [cad.quantities.total_quantity(weight_field).to(weight_units).v]
        )

        if stat_name == BulkAverage.name:
            tmp = cad.quantities.weighted_average_quantity(list(fields),
                                                           weight_field)
            rslt['average'] = np.empty((len(fields),), dtype = np.float64)
            for i in range(len(fields)):
                rslt['average'][i] = tmp[i].to(quantity_units).v
        else:
            tmp = cad.quantities.weighted_variance(list(fields), weight_field)
            rslt['variance'] = np.empty((len(fields),), dtype = np.float64)
            rslt['average'] = np.empty((len(fields),), dtype = np.float64)
            for i in range(len(fields)):
                # technically, tmp[i][0] holds standard deviation, NOT the
                # variance
                rslt['variance'][i] = (tmp[i][0].to(quantity_units).v)**2
                rslt['average'][i] = tmp[i][1].to(quantity_units).v
        out[0][cr_ind] = rslt

    return out

def compare_bulk_stat(stat_kw_pair, ds, force_subvols_per_ax = (1,1,1),
                      **compare_kwargs):
    # force_subvols_per_ax only affects the call to small_dist_sf_props 
    my_geometric_selector = BoxSelector(
        left_edge = [-4.0,-4.0,-4.0], right_edge = [4.0,4.0,4.0],
        length_unit = 'code_length',
    )

    component_fields = (('gas','velocity_x'),
                        ('gas','velocity_y'),
                        ('gas','velocity_z'))
    quantity_units = 'cm/s'

    bin_edges = np.array([5./18., 5./9., 5./6., 10./9., 25./18.])
    my_cut_regions = [f'obj["logX_K"].v < {float(bin_edges[1]):.15e}',
                      f'(obj["logX_K"].v >= {float(bin_edges[1]):.15e}) & ' +
                      f'(obj["logX_K"].v < {float(bin_edges[3]):.15e})']
    
    alt_rslt_l = _yt_calc_bulkstat(component_fields, ds, my_geometric_selector,
                                   my_cut_regions, [stat_kw_pair],
                                   quantity_units = quantity_units)

    tmp = small_dist_sf_props(
        ds,
        cut_regions = my_cut_regions,
        quantity_units = quantity_units,
        component_fields = component_fields,
        force_subvols_per_ax = force_subvols_per_ax,
        max_points = None,
        rand_seed = None,
        geometric_selector = my_geometric_selector,
        statistic = [stat_kw_pair[0]],
        kwargs = [stat_kw_pair[1]],
        pool = None,
        dist_bin_edges = [1,2,3], # dummy value
        pos_units = "cm", # unimportant
    )
    actual_rslt_l = tmp[0]

    bulkvariance_cmp = (stat_kw_pair[0] == BulkVariance.name)

    for i in range(len(my_cut_regions)):
        compare_bulkstat(alt_rslt_l[0][i], actual_rslt_l[0][i],
                         bulkvariance_cmp = bulkvariance_cmp, **compare_kwargs)


if __name__ == '__main__':
    ds = setup_ds()
    
    compare_bulk_stat(
        ('bulkaverage', {'weight_field' : ( ('gas', 'cell_mass'), 'g')}),
         ds, force_subvols_per_ax = (1,1,1),
        weight_total_rtol = 2e-16,
        average_rtol = 4e-16
    )

    compare_bulk_stat(
        ('bulkvariance', {'weight_field' : ( ('index', 'ones'), 'dimensionless')}),
         ds, force_subvols_per_ax = (1,1,1),
        weight_total_rtol = 2e-16,
        variance_rtol = 4e-16,
        average_rtol = 4e-16
    )

    compare_bulk_stat(
        ('bulkvariance', {'weight_field' : ( ('gas', 'cell_mass'), 'g')}),
         ds, force_subvols_per_ax = (1,1,1),
        weight_total_rtol = 2e-16,
        variance_rtol = 3e-16,
        average_rtol = 4e-16
    )

