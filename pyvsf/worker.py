from copy import deepcopy
from typing import Tuple, Sequence, NamedTuple, Dict, Any
import numpy as np

from .pyvsf import vsf_props

from ._kernels import get_kernel
from ._kernels_cy import build_consolidater
from ._perf import PerfRegions
from ._cut_region_iterator import get_cut_region_itr_builder

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

class StatDetails(NamedTuple):
    # lightweight class used internally by SFWorker

    # maps names of stats to the corresponding index:
    name_index_map: Dict[str, int]
    # stat_kw_pairs for all structure function statistics:
    sf_stat_kw_pairs: Sequence[Tuple[str, Dict[str,Any]]]
    # pairs of kernels and kwargs for non-structure function statistics:
    nonsf_kernel_kw_pairs: Sequence[Tuple[Any, Dict[str,Any]]]

def _process_stat_choices(stat_kw_pairs, sf_params):
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
    for stat_ind, (stat_name, kw) in enumerate(stat_kw_pairs):
        # 1. load the kernel
        kernels.append(get_kernel(stat_name))
        # 2. update extra_quan_spec (if necesary)
        try:
            tmp = kernels[stat_ind].get_extra_fields(kw, sf_params = sf_params)
        except TypeError:
            # not all implementations of get_extra_fields expect sf_params as a
            # keyword argument (this is a little bit of a hack)
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
            stat_details.nonsf_kernel_kw_pairs.append( (kernels[stat_ind], kw) )
    return kernels, extra_quan_spec, stat_details

class StatRsltContainer:
    def __init__(self, num_statistics, num_cut_regions):
        self.num_statistics = num_statistics
        self.num_cut_regions = num_cut_regions
        self._arr = np.empty((num_statistics, num_cut_regions),
                             dtype = object)

    @classmethod
    def build_empty_container(cls, num_statistics, num_cut_regions):
        out = cls(num_statistics = num_statistics,
                  num_cut_regions = num_cut_regions)
        for i in range(num_cut_regions):
            out.store_all_empty_cut_region(i)
        return out

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
        self.perf_region = deepcopy(perf_region)

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

_PERF_REGION_NAMES = ('all', 'auto-sf', 'auto-other', 'cross-sf', 'cross-other')

class _BaseWorker:
    """
    Encapsulates a basic worker that computes statistics based on some kind of
    kernels
    """

    def process_index(self, subvol_index, perf):
        raise NotImplementedError("This method must be implemented by "
                                  "subclasses")

    def __call__(self, subvol_indices):
        tmp = []
        for subvol_index in subvol_indices:
            try:
                tmp.append(self.process_index(subvol_index))
            except BaseException as e:
                raise RuntimeError(
                    f"Problem encountered while processing {subvol_index}"
                ) from e
        return tmp

class SFWorker(_BaseWorker):
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

    

    def process_index(self, subvol_index):
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
        kernels, extra_quan_spec, stat_details = _process_stat_choices(
            self.stat_kw_pairs, self.sf_param
        )

        any_cross_stats = (
            (len(stat_details.sf_stat_kw_pairs) > 0) or
            any(kernel.operate_on_pairs for kernel, _ \
                in stat_details.nonsf_kernel_kw_pairs)
        )

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
        neighbor_ind_iter = cut_region_itr_builder.neighbor_ind_iter
        for other_ind in neighbor_ind_iter(subvol_index, self.subvol_decomp):
            #print(f"{subvol_index}-{other_ind}")

            if any_cross_stats:
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
            else:
                cross_sf_rslts.append(
                    StatRsltContainer.build_empty_container(
                        num_statistics = self._get_num_statistics(),
                        num_cut_regions = self._get_num_cut_regions()
                ))

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
                          perf_region = perf)



