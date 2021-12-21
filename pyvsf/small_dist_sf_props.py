
from itertools import product
import logging
from typing import Tuple, Sequence, Optional

import numpy as np
from pydantic import (
    BaseModel,
    conlist,
    validator,
    PositiveInt,
    root_validator
)

from .pyvsf import vsf_props
from ._cut_region_iterator import (
    get_root_level_cell_width,
    neighbor_ind_iter,
    get_cut_region_itr_builder
)

from ._kernels import get_kernel

# Define some Data Objects

class BoxSelector(BaseModel): 
    left_edge: Tuple[float, float, float]
    right_edge: Tuple[float, float, float]
    length_unit: str

    class Config:
        allow_mutation = False

    def apply_selector(self, ds, **kwargs):
        left_edge = ds.arr(self.left_edge, self.length_unit)
        right_edge = ds.arr(self.right_edge, self.length_unit)
        return ds.box(left_edge = left_edge, right_edge = right_edge,
                      **kwargs)

    def get_bbox(self):
        return (np.array(self.left_edge), np.array(self.right_edge),
                self.length_unit)

_ytfield_type = Tuple[str, str]

def _validate_ytfield(val: _ytfield_type) -> _ytfield_type:
    if (not isinstance(val, tuple)) or len(val) != 2:
        raise ValueError('must be tuple of 2 vals')
    elif not (isinstance(val[0], str) and isinstance(val[1], str)):
        raise ValueError("each item must be a string")
    return val

def _validate_bin_edges(val: Sequence[float]) -> Sequence[float]:
    if len(val) < 2:
        raise ValueError("must have at least 2 entries")
    out = tuple(val)
    if np.any(np.diff(out) <= 0.0):
        raise ValueError("must monotonically increase")
    return out

class StructureFuncProps(BaseModel):
    dist_bin_edges: Sequence[float]
    dist_units: str
    quantity_components: conlist(_ytfield_type, min_items = 1, max_items = 3)
    quantity_units: str
    cut_regions: conlist(Optional[str], min_items = 1)
    max_points: Optional[PositiveInt] = ...
    geometric_selector: Optional[BoxSelector]

    # validators
    _validate_comp = validator('quantity_components', each_item = True,
                               allow_reuse=True)(_validate_ytfield)
    _validate_dist_bin_edges = validator('dist_bin_edges',
                                         allow_reuse=True)(_validate_bin_edges)

    class Config:
        allow_mutation = False

class SubVolumeDecomposition(BaseModel):
    left_edge: Tuple[float, float, float]
    right_edge: Tuple[float, float, float]
    length_unit: str
    subvols_per_ax: Tuple[PositiveInt, PositiveInt, PositiveInt]
    periodicity: Tuple[bool, bool, bool]

    class Config:
        allow_mutation = False

    @root_validator(pre = False)
    def check_edge(cls, values):
        for i in range(3):
            assert values['left_edge'][i] < values['right_edge'][i]
        return values

    @property
    def subvol_widths(self):
        l,r = np.array(self.left_edge), np.array(self.right_edge)
        return (r - l) / np.array(self.subvols_per_ax), self.length_unit

    def valid_subvol_index(self, subvol_index):
        if len(subvol_index) != 3:
            raise ValueError(f"Invalid subvol_index: {subvol_index}")
        itr = [0,1,2]
        return (
            all(int(subvol_index[i]) == subvol_index[i] for i in itr) and
            all(0 <= subvol_index[i] < self.subvols_per_ax[i] for i in itr)
        )





def decompose_volume(ds, sf_params, force_single_subvol = False):
    """
    Constructs an instance of SubVolumeDecomposition.

    Paramters
    ---------
    ds
        The dataset object
    sf_params: StructureFuncProps
        structure function parameters.

    Notes
    -----
    If we want to support calculation of structure function as a function of 
    position in the future (analagous to a STFT without overlap), it might be
    nice to support the following:
    - Specify an arbitrary subvolume width
    - let SubVolumeDecomposition support a nominal subvolume size for all
      subvolumes other than those adjacent to a boundary. Those adjacent 
      subvolumes could have a smaller width (for a non-periodic boundary). If 
      we did this, we would presumably want to center the standard sized 
      subvolumes on the remainder of the domain.
    """

    # it might be nice to be able to specify the max resolvable length in a
    # subvolume

    kwargs = {}

    if sf_params.geometric_selector is not None:
        left, right, len_u = sf_params.geometric_selector.get_bbox()
        kwargs['left_edge'] = tuple(left)
        kwargs['right_edge'] = tuple(right)
    else:
        len_u = str(ds.domain_left_edge.units)
        kwargs['left_edge'] = tuple(ds.domain_left_edge.v)
        kwargs['right_edge'] = tuple(ds.domain_right_edge.to(len_u).v)
    kwargs['length_unit'] = len_u

    width = ds.arr(
        np.array(kwargs['right_edge']) - np.array(kwargs['left_edge']), len_u
    )
    assert (width > 0).all()

    if force_single_subvol:
        kwargs['subvols_per_ax'] = (1,1,1)
    else:
        # retrieve the root-level cell_width in units of 'code_length'
        root_level_cell_width = get_root_level_cell_width(ds)

        # retrieve the max edge of a distance bin
        max_dist_bin_edge = ds.quan(np.amax(sf_params.dist_bin_edges),
                                    sf_params.dist_units)

        # if the edges of our subvolumes were always guaranteed to be aligned
        # with the edges of a root level cell, and there were no round-off
        # error, then the min_subvol_width should exactly be:
        #  > root_level_cell_width.to(sf_params.dist_units) + max_dist_bin_edge
        # If the subvolume widths were any smaller, then we could miss pairs of
        # points at a separation of max_dist_bin_edge
        #
        # Since we aren't taking the care to do that, we will double the
        # contribution from the root_cell_width
        cell_width = root_level_cell_width.to(sf_params.dist_units)
        if (cell_width < 1e-5*(cell_width + max_dist_bin_edge)).any():
            # this is meant as an indication that there could be a problem
            # (round-off errors might start dropping pairs). The threshold was
            # chosen arbitrarily (the relative size probably needs to be a lot
            # smaller than 1e-5)
            raise RuntimeError("consider adjusting the fudge factor")

        min_subvol_width = (max_dist_bin_edge + 2*cell_width)

        kwargs['subvols_per_ax'] = tuple(map(
            lambda x: max(1, int(x)),
            np.floor((width / min_subvol_width).to('dimensionless').v)
        ))

    # TODO fix periodicity handling
    # Note: ds.periodicity doesn't store the right values for EnzoPDatasets
    kwargs['periodicity'] = (False, False, False)
    return SubVolumeDecomposition(**kwargs)


class TaskResult:
    # the fact that we allow num_statistics to be more than 1 is a little
    # premature

    def __init__(self, subvol_index, main_subvol_available_points,
                 num_statistics, num_cut_regions):
        self.subvol_index = subvol_index
        tmp = np.array(main_subvol_available_points)
        if tmp.shape != (num_cut_regions,):
            raise ValueError("main_subvol_available_points should be a 1D "
                             "array with an entry for each cut_region")
        self.main_subvol_available_points = tmp
        # in the future, we may want to actually make the container for the
        # main_subvol_rslts and consolidated_rslts into its own class and
        # simplify this class
        self._main_subvol_rslts = np.empty((num_statistics, num_cut_regions),
                                           dtype = object)
        self._consolidated_rslts = np.empty((num_statistics, num_cut_regions),
                                            dtype = object)

    def store_result(self, stat_index, cut_region_index,
                     main_subvol_rslt, consolidated_rslt):
        if self._main_subvol_rslts[stat_index, cut_region_index] is not None:
            raise IndexError("A result has already been stored for "
                             f"stat_index = {stat_index}, "
                             f"cut_region_index = {cut_region_index}")
        elif main_subvol_rslt is None:
            # consolidated_rslt is allowed to be None
            raise IndexError("main_subvol_rslt can't be None")

        self._main_subvol_rslts[stat_index, cut_region_index] = main_subvol_rslt
        self._consolidated_rslts[stat_index, cut_region_index] = (
            consolidated_rslt
        )

    def retrieve_result(self, stat_index, cut_region_index):
        if self._main_subvol_rslts[stat_index, cut_region_index] is None:
            raise ValueError("There are no results to retrieve for "
                             f"stat_index = {stat_index}, "
                             f"cut_region_index = {cut_region_index}")
        return (self._main_subvol_rslts[stat_index, cut_region_index],
                self._consolidated_rslts[stat_index, cut_region_index])

    def entries_stored_for_all_results(self):
        return all((e is not None) for e in self._main_subvol_rslts.flat)



def consolidate_partial_vsf_results(statistic, *rslts):
    """
    This function is used to consolidate the partial results from multiple 
    executions of pyvsf's `vsf_props` function.
    """
    if len(rslts) == 0:
        raise RuntimeError()
    kernel = get_kernel(statistic)
    return kernel.consolidate_stats(*rslts)

class SFWorker:
    """
    Computes the structure function properties for different subvolumes
    """
    def __init__(self, ds_initializer, subvol_decomp, sf_param, statistic,
                 kwargs, eager_loading = False):
        self.ds_initializer = ds_initializer
        self.subvol_decomp = subvol_decomp
        if any(subvol_decomp.periodicity):
            raise RuntimeError("Can't currently handle periodic boundaries")
        self.sf_param = sf_param
        self.statistic = statistic
        self.kwargs = kwargs
        self.eager_loading = eager_loading

    def __call__(self, subvol_indices):
        tmp = []
        for subvol_index in subvol_indices:
            try:
                tmp.append(self._process_index(subvol_index))
            except BaseException as e:
                raise RuntimeError(
                    f"Problem encountered while processing {subvol_index}"
                ) from e
        return tmp

    def _process_index(self, subvol_index):
        ds = self.ds_initializer()

        assert self.subvol_decomp.valid_subvol_index(subvol_index)
        assert self.sf_param.max_points is None

        # cut_region_itr_builder constructs iterators over cut_regions for a
        # given subvolume_index
        cut_region_itr_builder = get_cut_region_itr_builder(
            ds, self.subvol_decomp, self.sf_param, rand_generator = None,
            eager_loader = self.eager_loading
        )

        sf_param = self.sf_param
        dist_bin_edges = np.copy(np.array(self.sf_param.dist_bin_edges))

        # define some lists that are used to store some data for the duration
        # of this method's evaluation
        main_subvol_available_points = []
        main_subvol_rslts = []
        main_subvol_pos_and_quan = []

        # First, load in the main assigned subvolume and compute the auto-vsf
        print(f"{subvol_index}-auto")
        itr = cut_region_itr_builder(subvol_index, is_central = True)

        for cut_region_ind, pos, quan, available_points in itr:
            main_subvol_available_points.append(available_points)
            main_subvol_pos_and_quan.append((pos,quan))

            if available_points <= 1:
                main_subvol_rslts.append({})
            else:
                main_subvol_rslts.append(
                    vsf_props(
                        pos_a = pos, vel_a = quan,
                        pos_b = None, vel_b = None,
                        dist_bin_edges = dist_bin_edges,
                        statistic = self.statistic,
                        kwargs = self.kwargs
                    )
                )

        cross_sf_rslts = []

        # Next, load the adjacent subvolumes (on the right side) and compute
        # cross term contributions

        for other_ind in neighbor_ind_iter(subvol_index, self.subvol_decomp):
            print(f"{subvol_index}-{other_ind}")

            cross_sf_rslts.append([])
            print(other_ind)

            itr = cut_region_itr_builder(other_ind, is_central = False)
            # iterate over the positions/quantities from the adjacent subvolume
            # for each cut region
            for cut_region_index, o_pos, o_quan, o_available_points in itr:
                # fetch the positions/quantities for the current cur region of
                # the main subvolume
                m_pos, m_quan = main_subvol_pos_and_quan[cut_region_index]
                m_available_points \
                    = main_subvol_available_points[cut_region_index]

                if (m_available_points == 0) or (o_available_points == 0):
                    cross_sf_rslts[-1].append({})
                else:
                    cross_sf_rslts[-1].append(
                        vsf_props(
                            pos_a = m_pos, vel_a = m_quan,
                            pos_b = o_pos, vel_b = o_quan,
                            dist_bin_edges = dist_bin_edges,
                            statistic = self.statistic,
                            kwargs = self.kwargs
                        )
                    )

        out = TaskResult(subvol_index, main_subvol_available_points,
                         num_statistics = 1,
                         num_cut_regions = len(sf_param.cut_regions))

        # finally, consolidate cross_sf_rslts together with main_subvol_rslts
        for cut_region_i, main_subvol_rslt in enumerate(main_subvol_rslts):
            consolidated_rslt = consolidate_partial_vsf_results(
                self.statistic, main_subvol_rslt,
                *[sublist[cut_region_i] for sublist in cross_sf_rslts]
            )
            out.store_result(stat_index = 0, cut_region_index = cut_region_i,
                             main_subvol_rslt = main_subvol_rslt,
                             consolidated_rslt = consolidated_rslt)

        assert out.entries_stored_for_all_results() # sanity check!!!
        return out

def subvol_index_batch_generator(subvol_decomp, n_workers,
                                 subvols_per_chunk = None):
    num_x, num_y, num_z = subvol_decomp.subvols_per_ax
    
    if subvols_per_chunk is None:
        if (n_workers == 1):
            chunksize = num_x
        elif n_workers % (num_y*num_z) == 0:
            chunksize = num_x
        else:
            num_subvols = num_x*num_y*num_z
            chunksize, remainder = divmod(num_subvols, 2*num_workers)
            if remainder != 0:
                chunksize+=1
            chunksize = min(chunksize, num_x)
    else:
        assert subvols_per_chunk <= num_x
        chunksize = subvols_per_chunk
    assert chunksize > 0

    cur_batch = []

    for z_ind in range(num_z):
        for y_ind in range(num_y):
            for x_ind in range(num_x):
                cur_batch.append((x_ind, y_ind, z_ind))
                if len(cur_batch) == chunksize:
                    yield tuple(cur_batch)
                    cur_batch = []
    if len(cur_batch) > 0:
        yield tuple(cur_batch)

_dflt_vel_components = (('gas','velocity_x'),
                        ('gas','velocity_y'),
                        ('gas','velocity_z'))

def small_dist_sf_props(ds_initializer, dist_bin_edges,
                        cut_regions = [None],
                        pos_units = None, quantity_units = None,
                        component_fields = _dflt_vel_components,
                        geometric_selector = None,
                        statistic = 'variance', kwargs = {},
                        max_points = None, rand_seed = None,
                        force_single_subvol = False,
                        eager_loading = False,
                        pool = None, autosf_subvolume_callback = None):
    """
    Computes the structure function.

    This function includes optimizations that excel the best when 
    `np.amax(dist_bin_edges)` is significantly smaller than the domain's width.
    This function avoids looking at a lot of pairs that are too far apart to 
    matter

    Suppose: 
    - `d` is `np.amax(dist_bin_edges)` times some small fudge factor 
       (e.g. ~1.001) in `pos_units`
    - `L`, `W`, `H`, are dimensions of the domain (in `pos_units`)
    - `n` is the number of points per 1 `pos_unit`.
    The brute-force approach that considers every pair considers:
        `0.5*(L*W*H)**2 * n**6` unique pairs
    The function instead considers roughly (13.5 is slightly too large):
        `13.5 * (L/d) * (W/d) * (H/d) * d**6 * n**6` pairs
    This means that the brute force approach considers a factor of 
    `L*W*H/(26 * d^3)` extra pairs

    Parameters
    ----------
    ds_initializer
        The callable that initializes a yt-dataset
    dist_bin_edges: 1D np.ndarray
        Optionally specifies the distance bin edges. A distance `dx` that falls
        in the `i`th bin satisfies the following inequality: 
        `dist_bin_edges[i] <= dx < dist_bin_edges[i+1]`
    cut_regions: tuple of strings
        `cut_regions` is list of cut_strings combinations. Examples 
        include `"obj['temperature'] > 1e6"`,
        `'obj["velocity_magnitude"].in_units("km/s") > 1'`, and None
    pos_units, quantity_units: string, Optional
        Optionally specifies the position and quantity units.
    component_fields: list of fields
        List of 1 to 3 `yt` fields that are used to represent the individual 
        components of the quntity for which the structure function properties
        are computed
    geometric_selector: BoxSelector, optional
        Optional specification of a subregion to compute the structure function
        within.
    maxpoints: int, optional
        The maximum number of points to consider in the calculation. When
        unspecified, there is no limit to the maximum number of points.
    rand_seed: int, optional
        Optional argument used to seed the pseudorandom permutation used to
        select points when the number of points exceed `maxpoints`.
    eager_loading: bool, optional
        When True, this tries to load simulation data much more eagerly
        (aggregating reads). While this requires additional RAM, this tries to
        ease pressure on shared file systems.
    pool: `multiprocessing.pool.Pool`-like object, optional
        When specified, this should have a `map` method with a similar
        interface to `multiprocessing.pool.Pool`'s `map method
        and an iterable.
    autosf_subvolume_callback: callable, Optional
        An optional callable that can process the auto-structure function
        properties computed for individual subvolumes (for example, this could
        be used to save such quantities to disk). The callable should expect
        the following arugments 
        - an instance of `SubVolumeDecomposition` (specifying how the domain is
          broken up). This should not be mutated.
        - the subvolume index (a tuple of 3 integers)
        - the structure function properties computed within the subvolume
        - the number of points in that subvolume that are available to be used
          to compute the structure function properties.

    Returns
    -------
    prop_l: list of dicts
        The properties dictionary for the entire domain (for each cut_region)
    num_points_used_arr: np.ndarray
        The total number of points that were used to compute prop_l (for each
        cut region)
    toal_avail_points_arr: np.ndarray
        The total number of points that are available to be used to compute
        structure function properties (for each cut region)
    subvol_decomp: `SubVolumeDecomposition`
        Specifies how the domain has been decomposed into subvolumes
    sf_params: StructureFuncProps
        Summarizes the structure function calculation properties

    """
    
    if not callable(ds_initializer):
        assert pool is None
        _ds = ds_initializer
        ds_initializer = lambda: _ds

    if pool is None:
        class Pool:
            def map(self, func, iterable, callback = None):
                tmp = map(func, iterable)
                for elem in tmp:
                    yield elem
        pool = Pool()
        n_workers = 1
    else:
        n_workers = pool.size

    assert len(cut_regions) > 0
    dist_bin_edges = np.asarray(dist_bin_edges, dtype = np.float64)
    if dist_bin_edges.ndim != 1:
        raise ValueError("dist_bin_edges must be a 1D np.ndarray")
    elif dist_bin_edges.size <= 1:
        raise ValueError("dist_bin_edges must have 2 or more elements")
    elif (dist_bin_edges[1:] <= dist_bin_edges[:-1]).any():
        raise ValueError(
            "dist_bin_edges must have monotonically increasing elements"
        )

    if (max_points is None) != (rand_seed is None):
        raise ValueError("max_points and rand_seed must both be "
                         "specified or unspecified")
    if max_points is not None:
        assert int(max_points) == max_points
        max_points = int(maxpoints)
        assert int(rand_seed) == rand_seed
        rand_seed = int(rand_seed)

        # to support this in the future, I think we need to adopt the
        # following procedure (for each cut_region)
        # 1. Root dispatches tasks where every subvolume counts up the number
        #    of valid points and send back to ro
        # 2. Root builds an array which lists the number of points per
        #    subvolume, availpts_per_subvol
        # 3. Root (it doesn't actually have to happen on root). Then randomly
        #    determines how many points come from each subvolume. Below is
        #    pseudo-code to sketch an inefficient way to do this:
        #      >>> gen = np.random.default_rng(seed = rand_seed - 1)
        #      >>> remaining = np.copy(availpts_per_subvol)
        #      >>> drawn = np.zeros_like(remaining)
        #      >>> for i in range(max_points):
        #      >>>     choice = gen.choice(remaining.size,
        #      ...                         p = remaining/remaining.sum())
        #      >>>     drawn[choice] += 1
        #      >>>     drawn[choice] -= 1
        #      >>> assert out.sum() == max_points
        #      >>> assert (remaining >= 0).all()
        # 4. Then, to identify the points for a subvolume, with index
        #    `sv_index`, use the following pseudo-code:
        #     >>> sv_ind1D = # 1D representation for sv_index
        #     >>> gen = np.random.default_rng(seed = rand_seed + sv_ind1D)
        #     >>> ipoints = gen.choice(availpts_per_subvol[sv_ind1D],
        #     ...                      size = drawn[choice], replace = False)

        raise NotImplementedError(
            "Support is not currently provided for randomly drawing a subset "
            "of points"
        )

    # some of the argument checking is automatically performed by validation in
    # structure_func_props
    structure_func_props = StructureFuncProps(
        dist_bin_edges = list(dist_bin_edges),
        dist_units = pos_units,
        quantity_components = component_fields,
        quantity_units = quantity_units,
        cut_regions = cut_regions,
        max_points = max_points,
        geometric_selector = geometric_selector
    )

    subvol_decomp = decompose_volume(ds_initializer(), structure_func_props,
                                     force_single_subvol = force_single_subvol)
    logging.info(
        f"Number of subvolumes per axis: {subvol_decomp.subvols_per_ax}"
    )
    worker = SFWorker(ds_initializer, subvol_decomp,
                      sf_param = structure_func_props,
                      statistic = statistic,
                      kwargs = kwargs,
                      eager_loading = eager_loading)

    iterable = subvol_index_batch_generator(subvol_decomp,
                                            n_workers = n_workers)
    result_l = [[] for i in cut_regions]
    total_num_points_arr = np.array([0 for i in cut_regions])

    for batched_result in pool.map(worker, iterable):
        for item in batched_result:
            subvol_index = item.subvol_index

            # subvol_available_pts is a lists of the available points from
            # just the subvolume at subvol_index (there is an entry for each
            # cut_region).
            subvol_available_pts = item.main_subvol_available_points

            # when statistic == "histogram", we can losslessly accumulate the
            # histograms as we receive them

            for stat_index in [0]:
                for cut_region_index in range(len(cut_regions)):

                    # for the given subvol_index, stat_index, cut_region_index:
                    # - main_subvol_rslts holds the contributions from just the
                    #   subvolume at subvol_index to the total structure
                    #   function
                    # - consolidated_rslt includes the contribution from
                    #   main_subvol_rslt as well as cross-term contributions
                    #   between points in subvol_index and points in its 13 (or
                    #   at least those that exist) nearest neigboring
                    #   subvolumes on the right side
                    main_subvol_rslt, consolidated_rslt \
                        = item.retrieve_result(stat_index, cut_region_index)

                    if autosf_subvolume_callback is not None:
                        autosf_subvolume_callback(subvol_decomp, subvol_index,
                                                  stat_index, cut_region_index,
                                                  main_subvol_rslts,
                                                  subvol_available_pts)
                    result_l[cut_region_index].append(consolidated_rslt)

            total_num_points_arr += subvol_available_pts
            print(subvol_index)
            print('num points from subvol: ', subvol_available_pts)
            print('total num points: ', total_num_points_arr)

    # now, let's consolidate the results together
    prop_l = [
        consolidate_partial_vsf_results(statistic, *sublist) \
        for sublist in result_l
    ]
    total_num_points_used_arr = np.array(total_num_points_arr)
    total_avail_points_arr = np.array(total_num_points_arr)
    return (prop_l, total_num_points_used_arr, total_avail_points_arr,
            subvol_decomp, structure_func_props)
    
    
