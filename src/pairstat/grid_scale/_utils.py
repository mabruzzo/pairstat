import numpy as np


def _top_level_grid_indices(ds):
    """
    Creates a 3D array holding the indices of the top level blocks for an
    Patch-based AMR dataset.

    Note: we use the term block and grid interchangably
    """
    assert ds.index.max_level == 0
    n_blocks = len(ds.index.grids)

    # there's probably a better way to do the following:
    root_block_width = (ds.index.grid_right_edge - ds.index.grid_left_edge)[0]
    to_nearest_int = lambda arr: np.trunc(arr + 0.5).astype(np.int32)

    blocks_per_axis = to_nearest_int((ds.domain_width / root_block_width).in_cgs().v)
    block_loc_array = np.empty(shape=blocks_per_axis, dtype=np.int32)
    assert block_loc_array.size == n_blocks

    block_array_indices = to_nearest_int(
        ((ds.index.grid_left_edge - ds.domain_left_edge) / root_block_width).in_cgs().v
    )

    block_loc_array[tuple(block_array_indices.T)] = np.arange(n_blocks)
    return block_loc_array


def get_compute_grid_spacing(block_index, ds):
    """
    Returns a tuple of spacing values

    We could be more exact about this (there might be round-off error)

    I think that we could dispose of this function.
    """
    tmp = _top_level_grid_indices(ds)
    return tuple(ds.index.grids[tmp[block_index]].dds.to("code_length").v)


def get_left_edge(block_index, ds, compute_grid_slc=None, cell_edge=True):
    """
    Returns the left-edge in code length units of the block

    When compute_grid_slc is not None, this actually computes the left edge of
    the compute grid.

    We could be more exact about this (there might be round-off error)

    Parameters
    ----------

    cell_edge: bool,optional
        When True (the default choice), this function returns the position of
        the left edge of the leftmost cell. Otherwise this function returns the
        position of the cell-center of the leftmost cell

    """

    tmp = _top_level_grid_indices(ds)

    data_block_left_edge = ds.index.grid_left_edge[tmp[block_index]].to("code_length").v

    if compute_grid_slc is None:
        if cell_edge:
            return data_block_left_edge
        else:
            raise NotImplementedError()
    else:
        spacing = get_compute_grid_spacing(block_index, ds)
        out = np.empty_like(data_block_left_edge)
        if cell_edge:
            offset = 0
        else:
            offset = 0.5
        for i in range(3):
            out[i] = (compute_grid_slc[i].start + offset) * spacing[
                i
            ] + data_block_left_edge[i]
        return out
