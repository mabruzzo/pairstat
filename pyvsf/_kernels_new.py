import functools

import numpy as np

from _kernels_cy import _allocate_unintialized_rslt_dict

class _NeighborOpExecutor:
    def __init__(self, axis, trailing_ghost):
        if trailing_ghost:
            default_slc = slice(None, -1)
        else:
            default_slc = slice(None)
        self.x_slcs = default_slc, default_slc
        self.y_slcs = default_slc, default_slc
        self.z_slcs = default_slc, default_slc
        
        if axis == 'x':
            self.x_slcs = slice(0,-1,1), slice(1,None, 1)
        elif axis == 'y':
            self.y_slcs =slice(0,-1,1), slice(1,None, 1)
        elif axis == 'z':
            self.z_slcs = slice(0,-1,1), slice(1,None, 1)

    def exec_on_array(self, op, arr):
        x_slc0, x_slc1 = self.x_slcs
        y_slc0, y_slc1 = self.y_slcs
        z_slc0, z_slc1 = self.z_slcs
        vals_0 = arr[x_slc0, y_slc0, z_slc0]
        vals_1 = arr[x_slc1, y_slc1, z_slc1]
        return op(vals_0, vals_1)

    def exec_on_field(self, op, field, grid):
        return self.exec_on_array(op, grid[field])

def _neighbor_vec_differences(components, axis, diff_type,
                              trailing_ghost = False):
    """
    Computes differences between vectors
    
    Parmeters
    ---------
    diff_type : str
        Accepts values of 'parallel', 'transverse', 'total'
    trailing_ghost: bool
        `True` indicates that there is a trailing ghost cell 
        at the end of each axis.
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
    exec_operation = _NeighborOpExecutor(axis, trailing_ghost)
    
    assert len(field_components) == 3

    if axis == 'x':
        aligned_comp = 0, (1,2)
    elif axis == 'y':
        aligned_comp,transverse_comps = 1, (2,0)
    elif axis == 'z':
        aligned_comp,transverse_comps = 2, (0,1)
    else:
        raise ValueError(f'invalid axis value: {axis}')

    op = lambda x0, x1: x1 - x0
    if diff_type == 'transverse':
        diff_l = [exec_operation.exec_on_field(op, comp, grid) \
                  for comp in transverse_comps]
        return np.sqrt(np.square(diff_l[0]) + np.square(diff_l[1]))
    elif diff_type == 'total':   # magnitude of vel including all 3 components
        diff_l = [exec_operation.exec_on_field(op, comp, grid) \
                  for comp in field_components]
        return np.sqrt(np.square(diff_l[0]) + np.square(diff_l[1]) +
                       np.square(diff_l[2]))
    elif diff_type == 'parallel': # signed difference of component parallel to 
                                  # displacement vector between 2 cells
        return exec_operation.exec_on_field(op, aligned_comp, grid)
    else:
        raise ValueError(f'invalid diff_type value: {diff_type}')

def neighbor_vdiffs(quan_dict, extra_quantities, cr_map, kwargs,
                    trailing_ghost = True):
    """
    Computes histograms of velocity differences between neigboring cells. The
    velocity differences are normalized by the maximum sound speed of each pair
    of cells.

    Parameters
    ----------
    quan_dict: np.ndarray
        Keys associated with 3D structured grids.
    extra_quan_dict: dict
        Keys associated with 3D structured grids.
    cr_map: dict
        The keys are indices of cut regions. The values are a  that can 
        be applied to a structured grid to select all the values in a given
        map.
    kwargs: dict
        This should be a 3-element dict. The keys should be 
        'aligned_vdiff_edges', 'transverse_vdiff_edges', and 'mag_vdiff_edges'.
        The values associted with each key should be 1D arrays holding 
        monotonically increasing velocity differences.
    """
    components = [quan_dict[("gas", "velocity_x")],
                  quan_dict[("gas", "velocity_y")],
                  quan_dict[("gas", "velocity_z")]]
    cs_vals = extra_quan_dict[('gas', 'sound_speed')]

    calc_pairs = [('aligned_vdiff_', 'parallel'),
                  ('transverse_vdiff_', 'transverse'),
                  ('mag_vdiff_', 'total')]
    
    out = {}
    for cr_ind in cr_map.keys():
        out[cr_ind] = GridscaleVdiffHistogram.zero_initialize_rslt(
            [], kwargs, False
        )
    for axis in 'xyz':
        exec_operation = _NeighborOpExecutor(axis, trailing_ghost)

        # find the maximum sound speed from pairs of cells
        max_shared_cs = exec_operation.exec_on_array(np.maximum, cs_vals)

        for prefix, diff_type in calc_pairs:
            dtype = f'{prefix}_counts'
            bin_edges = kwargs[f'{prefix}_edges']

            # find the velocity differences
            vdiffs = (_neighbor_vec_differences(components, axis, diff_type,
                                                trailing_ghost)
                      / max_shared_cs)

            for cr_ind, cr_select_mask in cr_map.keys():
                # identify pairs of of cells that are in the same cut region
                idx = exec_operation.exec_on_array(np.logical_and,
                                                   cr_select_mask)
                # bin the velocity counts
                out[cr_ind][dtype] += np.histogram(vdiffs[idx],
                                                   bins = bin_edges)[0]
    return out

class GridscaleVdiffHistogram:
    # the velocity differences are normalized by sound speed...

    name = "grid_vdiff_histogram"
    output_keys = ('aligned_vdiff_counts',
                   'transverse_vdiff_counts',
                   'mag_vdiff_counts')
    commutative_consolidate = True
    operate_on_pairs = True
    non_vsf_func = neighbor_vdiffs
    structured_grid_inputs = True

    @classmethod
    def n_ghost_ax_end(cls):
        return 1

    @classmethod
    def get_extra_fields(cls, kwargs = {}, sf_params = None):
        # passing sf_params is a hack!
        quan_components = sf_params.quantity_components
        assert ('gas', 'velocity_x') in quan_components
        assert ('gas', 'velocity_y') in quan_components
        assert ('gas', 'velocity_z') in quan_components
        assert len(quan_components) == 3
        return {('gas', 'sound_speed') : (sf_params.quan_units,
                                          cls.operate_on_pairs)}

    @classmethod
    def get_dset_props(cls, dist_bin_edges, kwargs = {}):
        assert len(kwargs) == 3
        out = []
        for prefix in ['aligned_vdiff_', 'transverse_vdiff_', 'mag_vdiff_']:
            shape = (kwargs[f'{prefix}_edges'].size - 1,)
            out.append((f'{prefix}_counts', np.int64, shape))
        return out

    @classmethod
    def consolidate_stats(cls, *rslts):
        raise NotImplementedError("This needs to be implemented")

    @classmethod
    def validate_rslt(cls, rslt, dist_bin_edges, kwargs = {}):
        _validate_basic_quan_props(cls, rslt, dist_bin_edges, kwargs)

    @classmethod
    def postprocess_rslt(cls, rslt):
        pass # do nothing

    @classmethod
    def zero_initialize_rslt(cls, dist_bin_edges, kwargs = {},
                             postprocess_rslt = True):
        rslt = _allocate_unintialized_rslt_dict(cls, dist_bin_edges, kwargs)
        for k in rslt.keys():
            rslt[k][:] = 0
        if postprocess_rslt:
            cls.postprocess_rslt(rslt)
        return rslt
