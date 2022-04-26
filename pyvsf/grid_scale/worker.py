from ..worker import (
    _BaseWorker,
    _PERF_REGION_NAMES,
    _process_stat_choices,
    StatRsltContainer,
    TaskResult
)

from gascloudtool.marching_cubes.mc_blocking_helper import (
    _top_level_grid_indices,
    get_left_edge
)

from .._perf import PerfRegions
from .._kernels import get_kernel

class WorkerStructuredGrid(_BaseWorker):
    """
    Computes the kernel properties from different subvolumes (which are 
    processed as structured subvolumes).

    In comparison to `SFWorker`, all cut_region data is loaded simultaneously
    (because the spatial structure is important). 

    The main idea for creating this is to handle cases where we just need 1 or 
    2 extra ghost zones from a neighboring block. This is useful for kernels:
        - that consider all pairs of direct neighbors
        - kernels that compute a mesh with marching cubes
        - kernels involving convolutions with small filters

    Parameters
    ----------
    ds_initializer
        Used to initialize a dataset
    subvol_decomp: SubVolumeDecomposition
        Specifies how the subvolume is decomposed
    sf_param: StructureFuncProps
        This is only used to track the quantity components and the cut_regions
    stat_kw_pairs: dict
        Specifies the names of stats that are to be computed and associated 
        keywords.
    """

    def __init__(self, ds_initializer, subvol_decomp, sf_param,
                 stat_kw_pairs):
        self.ds_initializer = ds_initializer
        self.subvol_decomp = subvol_decomp
        if any(subvol_decomp.periodicity):
            raise RuntimeError("Can't currently handle periodic boundaries")
        self.stat_kw_pairs = stat_kw_pairs
        self.sf_param = sf_param

        assert len(self.stat_kw_pairs) == 1

    def _get_num_statistics(self):
        return len(self.stat_kw_pairs)

    def _get_num_cut_regions(self):
        return len(self.sf_param.cut_regions)

    def _get_all_inclusive_cr_index(self):
        raise NotImplementedError("Not Implemented yet")

    @staticmethod
    def load_subvol_data(ds, subvol_index, extra_quan_spec, subvol_decomp,
                         sf_props, n_ghost_cells = 1):
        # load in the data as a covering_grid:
        if ds.index.max_level == 0:
            root_block_shape = tuple(
                int(elem) for elem in ds.index.grids[0].shape
            )
        else:
            raise RuntimeError('Not equipped to handle AMR')

        if subvol_decomp.instrinsic_decomp:
            block_index = subvol_index
        else:
            raise RuntimeError("unsure how to handle this case")

        filled_shape = []
        for ax in range(3):
            if (block_index[ax] + 1) > subvol_decomp.subvols_per_ax[ax]:
                raise RuntimeError("This should not happen")
            length = root_block_shape[ax]
            if (block_index[ax] + 1) != subvol_decomp.subvols_per_ax[ax]:
                length += n_ghost_cells
            filled_shape.append(length)

        grid = ds.covering_grid(
            level = 0,
            left_edge = get_left_edge(block_index, ds, cell_edge = True),
            dims = tuple(filled_shape)
        )

        # now load data from the grid

        # returns the index of the cut region and the selector for the
        # region
        cr_map = {}
        my_locals = {"obj" : grid}
        for i, condition in enumerate(sf_props.cut_region):
            if condition is None:
                yield i, np.ones(grid.shape, np.bool8)
            else:
                idx = eval(condition, my_locals)
                cr_map[i] = idx
        grid.clear_data()

        pos = {'x' : grid['index', 'x'], 'y' : grid['index', 'y'],
               'z' : grid['index', 'z']}
        quans = []
        for field in sf_props.quantity_components:
            quans.append(
                grid[field].to(sf_props.quantity_units).ndarray_view()
            )
        equan_dict = {}
        for equan_name, (equan_units, _) in extra_quantities.items():
            equan_dict[equan_name] = \
                cad[equan_name].to(equan_units).ndarray_view()

        grid.clear_data()
        return cr_map, pos, quans, equan_dict

    def process_index(self, subvol_index):
        perf = PerfRegions(_PERF_REGION_NAMES)
        perf.start_region('all')

        ds = self.ds_initializer()

        assert self.subvol_decomp.valid_subvol_index(subvol_index)

        # Handle some stuff related to the choice of statistics:
        kernels, extra_quan_spec, stat_details =  _process_stat_choices(
            stat_kw_pairs
        )

        assert len(stat_details.sf_stat_kw_pairs) == 0 # sanity check!

        load_subvol_data(ds, index, extra_quan_spec, subvol_decomp, sf_props)
        cr_map, pos, quans, equan_dict = load_subvol_data(
            ds, index, extra_quan_spec, self.subvol_decomp
        )

        

        # now actually compute the statistic
        rslts = StatRsltContainer(
            num_statistics = 1,
            num_cut_regions = len(cr_map),
        )
        main_subvol_available_points = np.zeros((self._get_num_cut_regions(),),
                                                dtype = np.int64)
        # TODO: in the future, come back & initialize each entry of
        # main_subvol_available_points appropriately. Need to be a little
        # careful about this since we include ghost zones

        with perf.region('auto-other'): # calc non structure-func stats

            for kernel, kw in stat_details.nonsf_kernel_kw_pairs:
                stat_index = stat_details.name_index_map[kernel.name]
                func = kernel.non_vsf_func
                rslt = func(quan = quan, extra_quantities = extra_quan,
                            kwargs = kw)

                for cr_ind, cr_ind_rslt in rslt.items():
                    rslt_container.store_result(stat_index = stat_index,
                                                cut_region_index = cr_ind,
                                                rslt = rslt)

        perf.stop_region('all')
        return TaskResult(subvol_index, main_subvol_available_points,
                          main_subvol_rslts, None,
                          num_neighboring_subvols = 0,
                          perf_region = deepcopy(perf))
