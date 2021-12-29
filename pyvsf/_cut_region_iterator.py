# module that defines some helper functions related to loading subvolume data
# in save_sf_to_file
import gc
from itertools import product
import logging

import numpy as np

def _is_grid_based(ds):
    # this is not exhaustive!
    if _is_grid_based:
        return ("enzop" in ds.fluid_types) or ("enzoe" in ds.fluid_types)
    return False

def neighbor_ind_iter(central_subvol_ind, subvol_decomp, yield_batches = False):
    """
    Generator that yields the indices of all valid neighboring subvolume indices
    for which the cross-structure function must be computed.

    yield_batches can be used to yield batches of neighboring indicies that are
    effectively organized into slices (this can be used for optimizing data 
    loading)
    """

    _neighbor_delta_ind_batches =(
        # case with dx = 1, dy = 0, dz =0
        ( ( 1, 0, 0), ),
        # cases with dy = 1, dz = 0
        ( (-1, 1, 0), ( 0, 1, 0), ( 1, 1, 0) ),
        # cases with dz = 1
        ( (-1,-1, 1), ( 0,-1, 1), ( 1,-1, 1),
          (-1, 0, 1), ( 0, 0, 1), ( 1, 0, 1),
          (-1, 1, 1), ( 0, 1, 1), ( 1, 1, 1))
    )

    is_valid= lambda ind: self.subvol_decomp.valid_subvol_index(ind)

    for b in _neighbor_delta_ind_batches:
        # determine all neigbor_ind in current batch
        tmp = []
        for (di, dj, dk) in b:
            neighbor_ind = (central_subvol_ind[0] + di,
                            central_subvol_ind[1] + dj,
                            central_subvol_ind[2] + dk)
            if subvol_decomp.valid_subvol_index(neighbor_ind):
                tmp.append(neighbor_ind)

        if len(tmp) == 0:
            continue
        elif yield_batches:
            yield tuple(tmp)
        else:
            for elem in tmp: yield elem

def _pos_quan_equan_arr_generator(data_region, sf_props, rand_generator = None,
                                  extra_quantities = {}):
    """
    Generator that yields the cut_region_index, positions, quantities, and
    extra quantities for each cut_region in data_region.

    Parameters
    ----------
    extra_quantities: dict
        The keys of extra_quantities should be names of fields, and the 
        corresponding entry should be a tuple where the first element specifies
        the desired units and the second entry is a boolean specifying if it's 
        used for pairs of points.
    """

    for cut_region_index, cut_string in enumerate(sf_props.cut_regions):

        if cut_string is None or cut_string == '':
            cad = data_region
        else:
            cad = data_region.cut_region(cut_string)

        # get the positions for each point
        pos = np.array([cad[ii].to(sf_props.dist_units).ndarray_view() \
                        for ii in ['x', 'y', 'z']])

        npoints = pos.shape[1]
        max_points = sf_props.max_points
        if npoints == 0:
            logging.warning("No points!")
            yield cut_region_index, None, None, None, 0
        else:
            if max_points is not None and npoints > max_points:
                assert rand_generator is not None
                logging.info(
                    f"Too many points (>{max_points}), discarding some"
                )
                ipoints \
                    = rand_generator.permutation(npoints)[:sf_props.max_points]
                pos = pos[:,ipoints]
            else:
                ipoints = slice(None)

            # try to be a little conservative about memory
            tmp_l = []
            for field in sf_props.quantity_components:
                tmp_l.append(cad[field][ipoints].to(sf_props.quantity_units)\
                             .ndarray_view())

            for i in range(len(tmp_l),3):
                tmp_l.append(np.zeros_like(tmp_l[0]))
            quan_arr = np.array(tmp_l)

            equan_dict = {}
            for equan_name, (equan_units, _) in extra_quantities.items():
                equan_dict[equan_name] = \
                    cad[equan_name].to(equan_units).ndarray_view()

            cad.clear_data()
            del ipoints
            yield cut_region_index, pos, quan_arr, equan_dict, npoints

    data_region.ds.index.clear_all_data()

def get_root_level_cell_width(ds):
    # level = 0 denotes the root level
    out = ds.index.select_grids(level = 0)[0].dds
    assert str(out.units) == 'code_length'
    return out

def _get_subvol_left_right_edges(ds, subvol_decomp, subvol_index):
    assert subvol_decomp.valid_subvol_index(subvol_index)
    subvol_widths, _ = subvol_decomp.subvol_widths
    ind = np.array(subvol_index)

    # compute the left and right edges of the subvolume
    subvol_left = np.array(subvol_decomp.left_edge) + ind * subvol_widths
    subvol_right = np.array(subvol_decomp.left_edge) + (ind+1) * subvol_widths

    # adjust subvol_right in cases where it's expected to line up with
    # subvol_decomp.right_edge
    for dim in range(3):
        if subvol_index[dim] == subvol_decomp.subvols_per_ax[dim]-1:
            subvol_right[dim] = subvol_decomp.right_edge[dim]

    # convert subvol_left and subvol_right to yt_arrays
    subvol_left = ds.arr(subvol_left, subvol_decomp.length_unit)
    subvol_right = ds.arr(subvol_right, subvol_decomp.length_unit)

    return subvol_left, subvol_right

def _build_padded_box_region(ds, left_edge, right_edge):
    lunit = left_edge.units
    runit = right_edge.units

    # first construct a padded box region, that will definitely include all
    # desired points and a little extra
    root_cell_width = get_root_level_cell_width(ds)
    lpadded = left_edge - root_cell_width.to(lunit)
    rpadded = right_edge + root_cell_width.to(runit)
    return ds.box(left_edge = lpadded, right_edge = rpadded)

def _build_subvolume_cutstr(left_edge, right_edge):
    # return a cutstring to select just the points in a subvolume

    lunit = left_edge.units
    runit = right_edge.units
    # now create a cut_region from box_region that discards all uneeded
    # points
    lval, rval = left_edge.ndarray_view(), right_edge.ndarray_view()
    parts = []
    for i,ax in enumerate('xyz'):
        parts.append(
            f'(obj["gas", "{ax}"].to("{lunit}").ndarray_view() >= {lval[i]})'
        )
        parts.append(
            f'(obj["gas", "{ax}"].to("{runit}").ndarray_view() <= {rval[i]})'
        )
    return ' & '.join(parts)

def subvolume_dataobjects(ds, subvol_indices, subvol_decomp):
    """
    Constructs an iterator of dataobjects for each subvolume indices. A given
    dataobjects only contain data from cells whose centers are located within
    the specified subvolume(s).

    When multiple sub_indices are specified, the union of their subvolumes
    should completely fill a rectangular prism. This facillitates aggregation
    the IO.

    Notes
    -----
    Naively using ds.box(left_edge = left_edge, right_edge = right_edge)
    will include all cells that have any overlap with the volume (and it
    seems to be subject to some roundoff errors). At the same time, I suspect
    that creating a cut_region will involve a lot of allocations (since the
    data needs to be compared everywhere). So we currently do a hybrid of this.

    We could probably do a lot better that building a cut_region, since we know
    the data spacing is regular. This speed difference is probably most
    significant when aggregating IO.
    """
    if isinstance(subvol_indices[0], int):
        subvol_indices = [subvol_indices]

    if len(subvol_indices) == 1:
        # make the box_region slightly bigger than the desired subvolume
        left_edge, right_edge = _get_subvol_left_right_edges(ds, subvol_decomp,
                                                             subvol_indices[0])
        box_region = _build_padded_box_region(ds, left_edge, right_edge)
        cut_str = _build_subvolume_cutstr(left_edge, right_edge)
        return [(subvol_indices[0], box_region.cut_region(cut_str))]

    else:
        # make box_region extend over all desired subvolumes

        # first, confirm that there are no gaps between the specified
        # subvolumes
        _inds = np.array(subvol_indices)
        _mins = _inds.min(axis=0)
        _maxes = _inds.max(axis=0)

        expected = frozenset(product(range(_mins[0],_maxes[0]+1),
                                     range(_mins[1],_maxes[1]+1),
                                     range(_mins[2],_maxes[2]+1)))
        if (len(subvol_indices) != len(expected) or
            frozenset(subvol_indices) != expected):
            raise RuntimeError("Something went wrong! There are gaps between "
                               "subvolume indices")

        # next, construct the box-region that contains all of the specified
        # subvolumes
        box_left_edge, _  = _get_subvol_left_right_edges(ds, subvol_decomp,
                                                         _mins)
        _, box_right_edge = _get_subvol_left_right_edges(ds, subvol_decomp,
                                                         _maxes)
        box_region = _build_padded_box_region(ds, box_left_edge,
                                              box_right_edge)

        # finally construct & return the iterator that gives the data_regions
        def _generator():
            for subvol_index in subvol_indices:
                left_edge, right_edge = _get_subvol_left_right_edges(
                    ds, subvol_decomp, subvol_index
                )
                cut_str = _build_subvolume_cutstr(left_edge, right_edge)
                yield subvol_index, box_region.cut_region(cut_str)
        return _generator()


class SimpleCutRegionIterBuilder:
    """
    Builder of cut_region iterators for specified subvolumes.

    The elements of the resulting iterator are meant to be used to compute
    structure function properties. Each element is a tuple with the elements:
        `(cut_region_index, pos, quan, npoints)`
    - `cut_region_index` specifies the index of the cut_region in 
      `sf_props.cut_regions`
    - `pos` and `quan` are arrays of positions and quantities in the cut-region
      that are to be used for calculating structure-function properties
    - `npoints` is the total number of points in the cut_region. In principle,
      the number of points actually included in `pos` and `quan` may be smaller
      based on the value of `sf_props.max_points`.

    Parameters
    ----------
    extra_quantities: dict
        The keys of extra_quantities should be names of fields, and the 
        corresponding entry should be a tuple where the first element specifies
        the desired units and the second entry is a boolean specifying if it's 
        used for pairs of points.


    TODO: when is_central == False, avoid loading unnecessary extra_quantities

    """
    def __init__(self, ds, subvol_decomp, sf_props, extra_quantities = {},
                 rand_generator = None):
        self.ds = ds
        self.subvol_decomp = subvol_decomp
        self.sf_props = sf_props
        self.extra_quantities = extra_quantities
        self.rand_generator = rand_generator

    def _pos_quan_equan_arr_iterator(self, data_region):
        return _pos_quan_equan_arr_generator(
            data_region, self.sf_props, self.rand_generator,
            extra_quantities = self.extra_quantities
        )

    def __call__(self, subvol_index, is_central = False):
        """
        Constructs the iterator for `subvol_index`.

        is_central is ignored (it's only included for compatibility with the 
        signatures of subclasses)
        """
        _, data_region = list(subvolume_dataobjects(self.ds, [subvol_index],
                                                    self.subvol_decomp))[0]
        assert _ == subvol_index # sanity check
        return self._pos_quan_equan_arr_iterator(data_region)


class EagerCutRegionIterBuilder(SimpleCutRegionIterBuilder):
    """
    Similar to SimpleCutRegionIterBuilder, but it eagerly loads all anticipated
    data at once and caches it in memory.

    This definitely requires more RAM per process, but it can be used to try to
    reduce stress on the shared file system (by aggregating reads). 

    It's not obvious why aggregating reads in time is better reduces stress on 
    the file system (it's possible that there's a misunderstanding on my part)

    Notes
    -----
    This eagerly evaluates the iterator for the main central subvolume and each
    of it's neighbors and stores the iterators for future use. You could 
    definitely be more strategic about all of this.

    TODO: avoid loading unnecessary extra_quantities
    """

    def __init__(self, *args, **kwargs):
        SimpleCutRegionIterBuilder.__init__(self, *args, **kwargs)
        # cached_iterators maps subvolume indices to a list of outputs from
        # _pos_quan_arr_generator 
        self.cached_iterators = {}
        self.cur_center = None

    def clear_cache(self, keep_cur_center = False, run_gc = True):
        if self.cur_center is None or not keep_cur_center:
            self.cached_iterators = {} # clear the full cache
            self.cur_center = None
        else:
            # remove the non-neighbors of cur_center
            unneeded_keys = set(self.cached_iterators.keys())
            unneeded_keys.remove(self.cur_center)
            for ind in neighbor_ind_iter(self.cur_center, self.subvol_decomp):
                unneeded_keys.remove(ind)
            for key in unneeded_keys:
                del self.cached_iterators[key]

        if run_gc:
            gc.collect()

    def _build_iterators_for_batch(self, index_batch):
        # constructs the cut_region iterators for each subvol index in
        # index_batch in a clever way that tries to minimize how many times
        # files are loaded
        indices = [e for e in index_batch if e not in self.cached_iterators]
        if len(indices) == 0:
            return

        index_region_pairs = subvolume_dataobjects(self.ds, index_batch,
                                                   self.subvol_decomp)
        # TODO:
        # we could get a lot more clever about how we store the preloaded data
        # (right now, we're being fairly wasteful)

        # now preload the iterators for each quantity
        for ind, data_region in index_region_pairs:
            assert ind not in self.cached_iterators
            # get the pos_quan_equan_arr_iterator for the current region
            _iterator = _pos_quan_equan_arr_generator(
                data_region, self.sf_props, self.rand_generator,
                extra_quantities = self.extra_quantities
            )

            # store the eagerly evaluated iterator
            self.cached_iterators[ind] = tuple(_iterator)

    def __call__(self, subvol_index, is_central = False):
        """
        Load the data object for subvol_index. is_central is ignored.
        """
        if is_central: # construct the cached data for the new iterator
            prev_center = self.cur_center
            self.cur_center = subvol_index
            if prev_center is not None:
                # clear out entries from the cache that aren't neighbors of the
                # new center
                self.clear_cache(self, keep_cur_center = True, run_gc = True)
                raise RuntimeError("Reusing this object for multiple central "
                                   "subvol indices is not tested yet")

            self._build_iterators_for_batch([self.cur_center])

            for batch in neighbor_ind_iter(self.cur_center, self.subvol_decomp,
                                           yield_batches = True):
                self._build_iterators_for_batch(batch)

        # retrieve the appropriate iterator from the cache
        assert subvol_index in self.cached_iterators
        return self.cached_iterators[subvol_index]

def get_cut_region_itr_builder(*args, eager_loader = False, **kwargs):
    if eager_loader:
        cls = EagerCutRegionIterBuilder
    else:
        cls = SimpleCutRegionIterBuilder
    return cls(*args, **kwargs)
    
