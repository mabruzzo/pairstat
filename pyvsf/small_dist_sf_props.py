
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import product
import logging
from typing import Tuple, Sequence, Optional, NamedTuple, Dict, Any

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

from ._kernels import get_kernel, kernel_operates_on_pairs
from ._kernels_cy import build_consolidater

from ._perf import PerfRegions

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


def _fmt_subvol_index(subvol_index):
    return f'({subvol_index[0]:2d}, {subvol_index[1]:2d}, {subvol_index[2]:2d})'



def decompose_volume(ds, sf_params, force_subvols_per_ax = None):
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

    if force_subvols_per_ax is not None:
        assert len(force_subvols_per_ax) == 3
        if any(int(e) != e for e in force_subvols_per_ax):
            raise ValueError(
                "force_subvols_per_ax includes a non-integer"
            )
        force_subvols_per_ax = tuple(int(e) for e in force_subvols_per_ax)
        if any(e <= 0 for e in force_subvols_per_ax):
            raise ValueError(
                "force_subvols_per_ax includes a non-positive integer"
            )

    if force_subvols_per_ax == (1,1,1):
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

        max_subvols_per_ax = tuple(map(
            lambda x: max(1, int(x)),
            np.floor((width / min_subvol_width).to('dimensionless').v)
        ))

        if force_subvols_per_ax is not None:
            for i in range(3):
                if max_subvols_per_ax[i] < force_subvols_per_ax[i]:
                    raise ValueError(
                        "Based on the parameters, the max number of "
                        f"subvols along axis {i} is {max_subvols_per_ax[i]}. "
                        f"The user requested {force_subvols_per_ax[i]}."
                    )
            kwargs['subvols_per_ax'] = force_subvols_per_ax
        else:
            kwargs['subvols_per_ax'] = max_subvols_per_ax

    # TODO fix periodicity handling
    # Note: ds.periodicity doesn't store the right values for EnzoPDatasets
    kwargs['periodicity'] = (False, False, False)
    return SubVolumeDecomposition(**kwargs)

class StatRsltContainer:
    def __init__(self, num_statistics, num_cut_regions):
        self.num_statistics = num_statistics
        self.num_cut_regions = num_cut_regions
        self._arr = np.empty((num_statistics, num_cut_regions),
                             dtype = object)

    def store_result(self, stat_index, cut_region_index, rslt):
        if self._arr[stat_index, cut_region_index] is not None:
            raise IndexError("A result has already been stored for "
                             f"stat_index = {stat_index}, "
                             f"cut_region_index = {cut_region_index}")
        elif rslt is None:
            raise ValueError("a result can't be None")
        else:
            self._arr[stat_index, cut_region_index] = rslt

    def store_all_empty_cut_region(self, cut_region_index):
        if any((e is not None) for e in self._arr[:, cut_region_index]):
            raise RuntimeError("At least one result in cut_region_index has "
                               "already been set")
        for i in range(self.num_statistics):
            self._arr[i, cut_region_index] = {}

    def duplicate_results_for_cut_region(self, src_cr_index, dest_cr_index):
        """
        Performs a deepcopy for all of the results for `src_cr_index` and
        stores them for `dest_cr_index`
        """
        assert src_cr_index != dest_cr_index
        for stat_ind in range(self.num_statistics):
            rslt = deepcopy(
                self.retrieve_result(stat_index = stat_ind,
                                     cut_region_index = src_cr_index)
            )
            self.store_result(stat_index = stat_ind,
                              cut_region_index = dest_cr_index,
                              rslt = rslt)

    def rslt_exists(self, stat_index, cut_region_index):
        return self._arr[stat_index, cut_region_index] is not None

    def retrieve_result(self, stat_index, cut_region_index):
        if self.rslt_exists(stat_index, cut_region_index):
            return self._arr[stat_index, cut_region_index]
        else:
            raise ValueError("There are no results to retrieve for "
                             f"stat_index = {stat_index}, "
                             f"cut_region_index = {cut_region_index}")

    def cut_region_iter(self, stat_index):
        assert all((e is not None) for e in self._arr[stat_index,:])
        return iter(self._arr[stat_index,:])

    def entries_stored_for_all_results(self):
        return all((e is not None) for e in self._arr.flat)

    def purge(self):
        self._arr[...] = None

class TaskResult:

    def __init__(self, subvol_index, main_subvol_available_points,
                 main_subvol_rslts, consolidated_rslts,
                 num_neighboring_subvols = None, perf_region = None):
        self.subvol_index = subvol_index
        tmp = np.array(main_subvol_available_points)
        if tmp.shape != (main_subvol_rslts.num_cut_regions,):
            raise ValueError("main_subvol_available_points should be a 1D "
                             "array with an entry for each cut_region")
        self.main_subvol_available_points = tmp

        assert main_subvol_rslts.entries_stored_for_all_results()
        self.main_subvol_rslts = main_subvol_rslts

        # entries in consolidated_rslts can conceivably be empty for some
        # statistics
        assert (main_subvol_rslts.num_statistics ==
                consolidated_rslts.num_statistics)
        assert (main_subvol_rslts.num_cut_regions ==
                consolidated_rslts.num_cut_regions)
        self.consolidated_rslts = consolidated_rslts

        # the following are not strictly necessary, but provide useful status
        # information
        self.num_neighboring_subvols = num_neighboring_subvols
        self.perf_region = perf_region

def consolidate_partial_vsf_results(statistic, *rslts,
                                    stat_kw = {}, dist_bin_edges = None):
    """
    This function is used to consolidate the partial results from multiple 
    executions of pyvsf's `vsf_props` function.
    """
    if len(rslts) == 0:
        raise RuntimeError()
    kernel = get_kernel(statistic)
    if dist_bin_edges is None:
        return kernel.consolidate_stats(*rslts)
    else:
        consolidator = build_consolidater(dist_bin_edges, kernel, stat_kw)
        return consolidator.consolidate(*rslts)

class MaxSizeCutRegionTracker:
    """
    Lightweight functor used to track which cut_region contains the maximum
    number of points.

    Notes
    -----
    This is primarily meant to assist with avoiding a duplicated calculation
    when you happen to have 2 cut_regions that include all points.
    """

    def __init__(self, ignore_cr_index = None):
        self._ignore_cr_index = ignore_cr_index
        self.max_size_cr_index, self._max_num_points = None, None

    def process_cr_size(self, cr_index, num_points):
        dont_ignore = cr_index != self._ignore_cr_index
        new_val_is_larger = ((self._max_num_points is None) or
                             (num_points > self._max_num_points))
        if dont_ignore and new_val_is_larger:
            self.max_size_cr_index, self._max_num_points = cr_index, num_points

    def matches_max_num_points(self, num_points):
        return ((self._max_num_points is not None) and
                (self._max_num_points == num_points))

class StatDetails(NamedTuple):
    # lightweight class used internally by SFWorker

    # maps names of stats to the corresponding index:
    name_index_map: Dict[str, int]
    # stat_kw_pairs for all structure function statistics:
    sf_stat_kw_pairs: Sequence[Tuple[str, Dict[str,Any]]]
    # pairs of kernels and kwargs for non-structure function statistics:
    nonsf_kernel_kw_pairs: Sequence[Tuple[Any, Dict[str,Any]]]


_PERF_REGION_NAMES = ('all', 'auto-sf', 'auto-other', 'cross-sf', 'cross-other')

class SFWorker:
    """
    Computes the structure function properties for different subvolumes
    """
    def __init__(self, ds_initializer, subvol_decomp, sf_param, stat_kw_pairs,
                 eager_loading = False):
        self.ds_initializer = ds_initializer
        self.subvol_decomp = subvol_decomp
        if any(subvol_decomp.periodicity):
            raise RuntimeError("Can't currently handle periodic boundaries")
        self.sf_param = sf_param
        self.stat_kw_pairs = stat_kw_pairs
        self.eager_loading = eager_loading

    def _get_num_statistics(self):
        return len(self.stat_kw_pairs)

    def _get_num_cut_regions(self):
        return len(self.sf_param.cut_regions)

    def _get_all_inclusive_cr_index(self):
        # return the cut_region index corresponding to an all inclusive cut
        # region (if there is one)
        try:
            all_inclusive_cr_ind = self.sf_param.cut_regions.index(None)
        except ValueError:
            all_inclusive_cr_ind = None
        return all_inclusive_cr_ind

    @staticmethod
    def process_auto_stats(cut_region_iter, stat_details, dist_bin_edges, perf,
                           rslt_container, available_points_arr,
                           pos_and_quan_cache_l,
                           all_inclusive_cr_index = None):
        """
        Computes the auto-component of stats from a single subvolume.

        Parameters
        ----------
        cut_region_iter
            Holds the data being iterated over
        rslt_container: StatRsltContainer
            Object where the statistic results are stored
        available_points_arr: 1D np.ndarray
            Array that will be updated with the number of available points per
            cut region
        pos_and_quan_cache_l
            list where tuples of the positions and quantities for each subregion
            will be cached (so they can be reused for computing cross-terms).
        all_inclusive_cr_index : int, optional
            Optionally specified cut_region_index corresponding to a cut_region
            that includes all points is specified. When specified and there is
            another cut_region that happens to also include all points in the
            subvolume, a duplicated calculation will be avoided.

        Notes
        -----
        While the optimization using all_inclusive_cr_index may not seem useful,
        for certain types of inputs (e.g. wind tunnel simulations at early
        times), this will offer some performance improvement
        """

        largest_cr_tracker = MaxSizeCutRegionTracker(
            ignore_cr_index = all_inclusive_cr_index
        )

        for tmp in cut_region_iter:
            cr_index, pos, quan, extra_quan, available_points = tmp

            available_points_arr[cr_index] = available_points
            pos_and_quan_cache_l.append((pos,quan))

            largest_cr_tracker.process_cr_size(cr_index, available_points)
            if ((cr_index == all_inclusive_cr_index) and
                largest_cr_tracker.matches_max_num_points(available_points)):

                # copy results from prior cut_region & skip the calculation
                rslt_container.duplicate_results_for_cut_region(
                    src_cr_index = largest_cr_tracker.max_size_cr_index,
                    dest_cr_index = cr_index
                )
                assert available_points > 0 # sanity check
                continue

            with perf.region('auto-sf'): # calc structure-func stats

                if len(stat_details.sf_stat_kw_pairs) != 0:
                    if (available_points <= 1):
                        rslts = [{} for _ in stat_details.sf_stat_kw_pairs]
                    else:
                        rslts = vsf_props(
                            pos_a = pos, vel_a = quan,
                            pos_b = None, vel_b = None,
                            dist_bin_edges = dist_bin_edges,
                            stat_kw_pairs = stat_details.sf_stat_kw_pairs,
                            postprocess_stat = False,
                            nproc = 1
                        )

                    itr = zip(rslts, stat_details.sf_stat_kw_pairs)
                    for rslt, (stat_name, _) in itr:
                        rslt_container.store_result(
                            stat_index = stat_details.name_index_map[stat_name],
                            cut_region_index = cr_index, rslt = rslt
                        )

            with perf.region('auto-other'): # calc non structure-func stats

                for kernel, kw in stat_details.nonsf_kernel_kw_pairs:
                    stat_index = stat_details.name_index_map[kernel.name]
                    if ( (available_points == 0) or
                         (kernel.operate_on_pairs and (available_points<=1)) ):
                        rslt = {}
                    else:
                        func = kernel.non_vsf_func
                        rslt = func(quan = quan, extra_quantities = extra_quan,
                                    kwargs = kw)
                    rslt_container.store_result(stat_index = stat_index,
                                                cut_region_index = cr_index,
                                                rslt = rslt)

    @staticmethod
    def process_cross_stats(cut_region_iter, main_subvol_pos_and_quan,
                            main_subvol_available_points, stat_details,
                            dist_bin_edges, perf, rslt_container,
                            all_inclusive_cr_index = None):
        """
        Parameters
        ----------
        all_inclusive_cr_index : int, optional
            Optionally specified cut_region_index corresponding to a cut_region
            that includes all points is specified. When specified and there is
            another cut_region that happens to also include all points in both
            subvolumes, a duplicated calculation will be avoided.

        Notes
        -----
        While the optimization using all_inclusive_cr_index may not seem useful,
        for certain types of inputs (e.g. wind tunnel simulations at early
        times), this will offer some performance improvement
        """

        largest_cr_tracker = MaxSizeCutRegionTracker(
            ignore_cr_index = all_inclusive_cr_index
        )

        # iterate over the positions/quantities/extra_quantities from the
        # adjacent subvolume for each cut region
        for cr_index,o_pos,o_quan,o_eq,o_available_points in cut_region_iter:

            # fetch the cached positions/quantities for the current cut_region
            # of the main subvolume
            m_pos, m_quan = main_subvol_pos_and_quan[cr_index]
            m_available_points = main_subvol_available_points[cr_index]

            if (m_available_points == 0) or (o_available_points == 0):
                rslt_container.store_all_empty_cut_region(cr_index)
                continue

            npoint_pair = (m_available_points, o_available_points)
            largest_cr_tracker.process_cr_size(cr_index, npoint_pair)
            if ((cr_index == all_inclusive_cr_index) and
                largest_cr_tracker.matches_max_num_points(npoint_pair)):
                # copy results from prior cut_region & skip the calculation
                rslt_container.duplicate_results_for_cut_region(
                    src_cr_index = largest_cr_tracker.max_size_cr_index,
                    dest_cr_index = cr_index
                )
                continue

            with perf.region('cross-sf'): # calc structure-func stats
                if len(stat_details.sf_stat_kw_pairs) != 0:
                    rslts = vsf_props(
                        pos_a = m_pos, vel_a = m_quan,
                        pos_b = o_pos, vel_b = o_quan,
                        dist_bin_edges = dist_bin_edges,
                        stat_kw_pairs = stat_details.sf_stat_kw_pairs,
                        postprocess_stat = False,
                        nproc = 0 # fall back to OMP_NUM_THREADS env var
                    )

                    itr = zip(rslts, stat_details.sf_stat_kw_pairs)
                    for rslt, (stat_name, _) in itr:
                        rslt_container.store_result(
                            stat_index = stat_details.name_index_map[stat_name],
                            cut_region_index = cr_index, rslt = rslt
                        )

            with perf.region('cross-other'): # calc non structure-func stats
                for kernel, kw in stat_details.nonsf_kernel_kw_pairs:
                    stat_index = stat_details.name_index_map[kernel.name]
                    if not kernel.operate_on_pairs:
                        rslt_container.store_result(stat_index = stat_index,
                                                    cut_region_index = cr_index,
                                                    rslt = {})
                    else:
                        raise NotImplementedError()

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
        perf = PerfRegions(_PERF_REGION_NAMES)
        perf.start_region('all')

        ds = self.ds_initializer()

        assert self.subvol_decomp.valid_subvol_index(subvol_index)
        assert self.sf_param.max_points is None

        # Handle some stuff related to the choice of statistics:
        # 1. load the appropriate kernel object
        # 2. build dictionary specifying extra fields that need to be loaded
        # 3. group the statistics based on whether they're related to structure
        #    functions, since its MUCH faster to compute multiple structure
        #    function statistics at once (especially for large problems)
        kernels = []
        extra_quan_spec = {}
        stat_details = StatDetails(name_index_map = {},
                                   sf_stat_kw_pairs = [],
                                   nonsf_kernel_kw_pairs = [])
        for stat_ind, (stat_name, kw) in enumerate(self.stat_kw_pairs):
            # 1. load the kernel
            kernels.append(get_kernel(stat_name))
            # 2. update extra_quan_spec (if necesary)
            tmp = kernels[stat_ind].get_extra_fields(kw)
            if (tmp is not None) and (len(extra_quan_spec) == 0):
                extra_quan_spec = tmp
            elif tmp is not None:
                raise NotImplementedError("Come back to this eventually")

            # 3. update stat_details
            stat_details.name_index_map[stat_name] = stat_ind
            if kernels[stat_ind].non_vsf_func is None:
                stat_details.sf_stat_kw_pairs.append( (stat_name, kw) )
            else:
                stat_details.nonsf_kernel_kw_pairs.append( (kernels[stat_ind],
                                                            kw) )
        # cut_region_itr_builder constructs iterators over cut_regions for a
        # given subvolume_index
        cut_region_itr_builder = get_cut_region_itr_builder(
            ds, self.subvol_decomp, self.sf_param, rand_generator = None,
            eager_loader = self.eager_loading,
            extra_quantities = extra_quan_spec
        )

        all_inclusive_cr_index = self._get_all_inclusive_cr_index()

        sf_param = self.sf_param
        dist_bin_edges = np.copy(np.array(self.sf_param.dist_bin_edges))

        # define some lists that are used to store some data for the duration
        # of this method's evaluation
        main_subvol_rslts = StatRsltContainer(
            num_statistics = self._get_num_statistics(),
            num_cut_regions = self._get_num_cut_regions()
        )
        main_subvol_available_points = np.zeros((self._get_num_cut_regions(),),
                                                dtype = np.int64)
        main_subvol_pos_and_quan = []

        # First, load in the main assigned subvolume and compute the auto-vsf
        # terms and terms of other statistics (that don't operate on pairs)
        #print(f"{subvol_index}-auto")
        SFWorker.process_auto_stats(
            cut_region_itr_builder(subvol_index, is_central = True),
            stat_details, dist_bin_edges, perf,
            rslt_container = main_subvol_rslts,
            available_points_arr = main_subvol_available_points,
            pos_and_quan_cache_l = main_subvol_pos_and_quan,
            all_inclusive_cr_index = all_inclusive_cr_index
        )

        assert main_subvol_rslts.entries_stored_for_all_results() # sanity check

        cross_sf_rslts = []

        # Next, load the adjacent subvolumes (on the right side) and compute
        # the cross term for the vsf (and any other stats)

        for other_ind in neighbor_ind_iter(subvol_index, self.subvol_decomp):
            #print(f"{subvol_index}-{other_ind}")

            cross_sf_rslts.append(StatRsltContainer(
                num_statistics = self._get_num_statistics(),
                num_cut_regions = self._get_num_cut_regions()
            ))

            SFWorker.process_cross_stats(
                cut_region_itr_builder(other_ind, is_central = False),
                main_subvol_pos_and_quan, main_subvol_available_points,
                stat_details, dist_bin_edges, perf,
                rslt_container = cross_sf_rslts[-1],
                all_inclusive_cr_index = all_inclusive_cr_index
            )

        # finally, consolidate cross_sf_rslts together with main_subvol_rslts
        consolidated_rslts = StatRsltContainer(
            num_statistics = self._get_num_statistics(),
            num_cut_regions = self._get_num_cut_regions()
        )

        
        for stat_ind, (stat_name, stat_kw) in enumerate(self.stat_kw_pairs):
            itr = enumerate(
                main_subvol_rslts.cut_region_iter(stat_index = stat_ind)
            )
            for cut_region_i, main_subvol_rslt in itr:
                if kernels[stat_ind].operate_on_pairs:
                    consolidated_rslt = consolidate_partial_vsf_results(
                        stat_name, main_subvol_rslt,
                        *[sublist.retrieve_result(stat_ind, cut_region_i) \
                          for sublist in cross_sf_rslts],
                        stat_kw = stat_kw, dist_bin_edges = dist_bin_edges
                    )
                else:
                    consolidated_rslt = deepcopy(main_subvol_rslt)
                consolidated_rslts.store_result(stat_index = stat_ind,
                                                cut_region_index = cut_region_i,
                                                rslt = consolidated_rslt)
        perf.stop_region('all')

        return TaskResult(subvol_index, main_subvol_available_points,
                          main_subvol_rslts, consolidated_rslts,
                          num_neighboring_subvols = len(cross_sf_rslts),
                          perf_region = deepcopy(perf))

def subvol_index_batch_generator(subvol_decomp, n_workers,
                                 subvols_per_chunk = None,
                                 max_subvols_per_chunk = None):

    num_x, num_y, num_z = subvol_decomp.subvols_per_ax
    
    if subvols_per_chunk is None:
        if (n_workers == 1):
            chunksize = num_x
        elif n_workers % (num_y*num_z) == 0:
            chunksize = num_x
        else:
            num_subvols = num_x*num_y*num_z
            chunksize, remainder = divmod(num_subvols, 2*n_workers)
            if remainder != 0:
                chunksize+=1
            chunksize = min(chunksize, num_x)
    else:
        assert subvols_per_chunk <= num_x
        chunksize = subvols_per_chunk
    assert chunksize > 0

    if max_subvols_per_chunk is not None:
        if ((subvols_per_chunk is not None) and
            (subvols_per_chunk > max_subvols_per_chunk)):
            raise ValueError("subvols_per_chunk can't exceed "
                             "max_subvols_per_chunk")
        elif int(max_subvols_per_chunk) != max_subvols_per_chunk:
            raise ValueError("max_subvols_per_chunk must be an integer")
        elif max_subvols_per_chunk <= 0:
            raise ValueError("max_subvols_per_chunk must be positive")

        chunksize = min(chunksize, max_subvols_per_chunk)

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

class _PoolCallback:
    def __init__(self, stat_kw_pairs, n_cut_regions, subvol_decomp,
                 dist_bin_edges, autosf_subvolume_callback,
                 structure_func_props):
        # the following are constants:
        self.stat_kw_pairs = stat_kw_pairs
        self.n_cut_regions = n_cut_regions
        self.subvol_decomp = subvol_decomp
        self.dist_bin_edges = dist_bin_edges
        self.autosf_subvolume_user_callback = autosf_subvolume_callback
        self.structure_func_props = structure_func_props
        self.total_count = np.prod(subvol_decomp.subvols_per_ax)

        # the following attributes are updated with each call
        self.tmp_result_arr = np.empty(
            shape = (len(stat_kw_pairs), n_cut_regions,
                     np.prod(subvol_decomp.subvols_per_ax)),
            dtype = object
        )
        self.total_num_points_arr = np.array([0 for _ in range(n_cut_regions)])

        self.accum_rslt = {}
        for stat_ind, (stat_name,_) in enumerate(stat_kw_pairs):
            if get_kernel(stat_name).commutative_consolidate:
                self.accum_rslt[stat_ind] = [{} for _ in range(n_cut_regions)]

        self.cumulative_count = -1
        self.cumulative_perf = PerfRegions(_PERF_REGION_NAMES)


    def __call__(self, batched_result):
        subvols_per_ax = self.subvol_decomp.subvols_per_ax
        autosf_subvolume_callback = self.autosf_subvolume_user_callback

        for item in batched_result:
            subvol_index = item.subvol_index

            subvol_index_1D = (
                subvol_index[0] +
                subvols_per_ax[0] * ( subvol_index[1] +
                                      subvols_per_ax[1] * subvol_index[2] )
            )

            # subvol_available_pts is a lists of the available points from
            # just the subvolume at subvol_index (there is an entry for each
            # cut_region).
            subvol_available_pts = item.main_subvol_available_points

            main_subvol_rslts = item.main_subvol_rslts
            consolidated_rslts = item.consolidated_rslts

            for stat_ind, (stat_name, stat_kw) in enumerate(self.stat_kw_pairs):
                kernel = get_kernel(stat_name)
                for cut_region_i in range(self.n_cut_regions):

                    # for the given subvol_index, stat_index, cut_region_index:
                    # - main_subvol_rslts holds the contributions from just the
                    #   subvolume at subvol_index to the total structure
                    #   function
                    # - consolidated_rslt includes the contribution from
                    #   main_subvol_rslt as well as cross-term contributions
                    #   between points in subvol_index and points in its 13 (or
                    #   at least those that exist) nearest neigboring
                    #   subvolumes on the right side
                    consolidated_rslt = consolidated_rslts.retrieve_result(
                        stat_ind, cut_region_i
                    )

                    if stat_ind in self.accum_rslt:
                        # in the case, consolidation of the statistic is
                        # commutative
                        self.accum_rslt[stat_ind][cut_region_i] = \
                            consolidate_partial_vsf_results(
                                stat_name,
                                self.accum_rslt[stat_ind][cut_region_i],
                                consolidated_rslt, stat_kw = stat_kw,
                                dist_bin_edges = self.dist_bin_edges
                            )
                    else:
                        self.tmp_result_arr[stat_ind, cut_region_i,
                                            subvol_index_1D] = consolidated_rslt

                    if autosf_subvolume_callback is not None:
                        main_subvol_rslt = main_subvol_rslts.retrieve_result(
                            stat_ind, cut_region_i
                        )
                        tmp = deepcopy(main_subvol_rslt)
                        kernel.postprocess_rslt(tmp)

                        autosf_subvolume_callback(
                            self.structure_func_props, self.subvol_decomp,
                            subvol_index, stat_ind, cut_region_i,
                            tmp, subvol_available_pts[cut_region_i]
                        )

            # we only update total_num_points_arr once per task rslt
            self.total_num_points_arr[:] += subvol_available_pts

            _str_prefix = f'Driver: {_fmt_subvol_index(subvol_index)} - '
            self.cumulative_count += 1
            self.cumulative_perf = self.cumulative_perf + item.perf_region

            template = (
                ("{_str_prefix} subvol #{cum_count} of {total_count} " +
                 "({n_neighbors:2d} neigbors)\n") +
                "{pad}perf-sec - {perf_summary}\n" +
                "{pad}num points from subvol: {subvol_available_pts}\n" +
                "{pad}total num points: {total_num_points_arr}"
            )

            print(template.format(
                _str_prefix = _str_prefix, pad = '    ',
                cum_count = self.cumulative_count,
                total_count = self.total_count,
                n_neighbors = item.num_neighboring_subvols,
                perf_summary = item.perf_region.summarize_timing_sec(),
                subvol_available_pts = subvol_available_pts,
                total_num_points_arr = self.total_num_points_arr
            ))

            item.main_subvol_rslts.purge()
            item.consolidated_rslts.purge()

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
                        force_subvols_per_ax = None,
                        eager_loading = False,
                        max_subvols_per_chunk = None,
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
        `'obj["velocity_magnitude"].in_units("km/s") > 1'`, and `None`. `None`
        includes all values. A minor optimization can be made if `None` is
        passed as the last tuple entry in subvolumes where another cut_region
        also includes all of the entries in the subvolume.
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
    max_subvols_per_chunk: int, optional
        The subvolumes are passed to the pool in chunks. This is used to
        optionally specify the maximum number of subvols that are included in a
        chunk.
    pool: `multiprocessing.pool.Pool`-like object, optional
        When specified, this should have a `map` method with a similar
        interface to `multiprocessing.pool.Pool`'s `map method
        and an iterable.
    autosf_subvolume_callback: callable, Optional
        An optional callable that can process the auto-structure function
        properties computed for individual subvolumes (for example, this could
        be used to save such quantities to disk). The callable should expect
        the following arugments
        - an instance of `StructureFuncProps`. This should not be mutated.
        - an instance of `SubVolumeDecomposition` (specifying how the domain is
          broken up). This should not be mutated.
        - the subvolume index (a tuple of 3 integers),
        - stat_index, the index corresponding to the statitic being computed.
          (If you're only computing a single statistic, this will always be 0)
        - the index corresponding to the cut_region
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
    total_avail_points_arr: np.ndarray
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
                    if callback is not None:
                        callback(elem)
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

    subvol_decomp = decompose_volume(
        ds_initializer(), structure_func_props,
        force_subvols_per_ax = force_subvols_per_ax
    )

    logging.info(
        f"Number of subvolumes per axis: {subvol_decomp.subvols_per_ax}"
    )

    if isinstance(statistic, str):
        if not isinstance(kwargs, dict):
            raise ValueError("kwargs must be a dict when statistic is a string")
        stat_kw_pairs = [(statistic, kwargs)]
        single_statistic = True
    elif len(statistic) == 0:
        raise ValueError("statistic can't be an empty sequence")
    elif not all(isinstance(e, str) for e in statistic):
        raise TypeError("statistic must be a string or a sequence of strings")
    elif isinstance(kwargs,dict) or not all(isinstance(e,dict) for e in kwargs):
        raise ValueError("When statistic is a sequence of strings, kwargs "
                         "must be a sequence of dicts.")
    elif len(statistic) != len(kwargs):
        raise ValueError("When statistic is a sequence of strings, kwargs "
                         "must be a sequence of as many dicts.")
    elif np.unique(statistic).size != len(statistic):
        raise ValueError("When statistic is a sequence of strings, none of "
                         "the strings are allowed to be duplicates.")
    else:
        stat_kw_pairs = list(zip(statistic, kwargs))
        single_statistic = False

    del statistic, kwargs # deleted for debugging purposes

    worker = SFWorker(ds_initializer, subvol_decomp,
                      sf_param = structure_func_props,
                      stat_kw_pairs = stat_kw_pairs,
                      eager_loading = eager_loading)

    iterable = subvol_index_batch_generator(
        subvol_decomp, n_workers = n_workers,
        max_subvols_per_chunk = max_subvols_per_chunk
    )

    post_proc_callback = _PoolCallback(
        stat_kw_pairs, n_cut_regions = len(cut_regions),
        subvol_decomp = subvol_decomp, dist_bin_edges = dist_bin_edges,
        autosf_subvolume_callback = autosf_subvolume_callback,
        structure_func_props = structure_func_props
    )

    for batched_result in pool.map(worker, iterable,
                                   callback = post_proc_callback):
        continue # simply consume the iterator

    print("Cumulative subvol-processing perf-sec -\n    " +
          post_proc_callback.cumulative_perf.summarize_timing_sec())

    # now, let's consolidate the results together
    prop_l = []
    for stat_ind, (stat_name,_) in enumerate(stat_kw_pairs):
        if stat_ind in post_proc_callback.accum_rslt:
            # the results for this stat are already consolidated
            prop_l.append(post_proc_callback.accum_rslt[stat_ind])
        else:
            tmp = []
            for sublist in post_proc_callback.tmp_result_arr[stat_ind]:
                tmp.append(consolidate_partial_vsf_results(
                    stat_name, *sublist, dist_bin_edges = dist_bin_edges
                ))
            prop_l.append(tmp)

        kernel = get_kernel(stat_name)
        for elem in prop_l[-1]:
            kernel.postprocess_rslt(elem)

    if single_statistic:
        prop_l = prop_l[0]
    total_num_points_used_arr \
        = np.array(post_proc_callback.total_num_points_arr)
    total_avail_points_arr \
        = np.array(post_proc_callback.total_num_points_arr)
    return (prop_l, total_num_points_used_arr, total_avail_points_arr,
            subvol_decomp, structure_func_props)
