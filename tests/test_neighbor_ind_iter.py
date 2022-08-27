import collections
import itertools
import functools

from pyvsf._cut_region_iterator import _neighbor_ind_iter
from pyvsf.small_dist_sf_props import SubVolumeDecomposition

def find_all_pairs(nx, ny, nz, get_neighbor_iter):
    """
    Returns unsorted list of pairs and a Counter denoting how many neighbors a
    given index has
    """
    subvol_decomp = SubVolumeDecomposition(
        left_edge = (0.0, 0.0, 0.0),
        right_edge = (10.0 * nx, 10.0* ny, 10.0*nz),
        length_unit = 'cm',
        subvols_per_ax = (nx, ny, nz),
        periodicity = (False, False, False),
        intrinsic_decomp = False
    )

    pair_l = []
    num_neighbor_l = []
    for ind in itertools.product(range(nx), range(ny), range(nz)):
        cur_num_neighbors = 0
        for neighbor_ind in get_neighbor_iter(ind, subvol_decomp):
            if ind < neighbor_ind:
                pair = (ind, neighbor_ind)
            else:
                pair = (neighbor_ind, ind)
            pair_l.append(pair)
            cur_num_neighbors+=1
        num_neighbor_l.append(cur_num_neighbors)
    return pair_l, collections.Counter(num_neighbor_l)


_traditional_get_neighbor_iter = functools.partial(
    _neighbor_ind_iter,
    force_traditional_neighbors = True
)

_alt_get_neighbor_iter = _neighbor_ind_iter

def test_all_neighbors(nx,ny,nz):

    ref_l, ref_counts = find_all_pairs(nx,ny,nz, _traditional_get_neighbor_iter)
    test_l, test_counts = find_all_pairs(nx,ny,nz, _alt_get_neighbor_iter)
    # no need to compare test_counts and ref_counts. They are just provided to
    # inform us about how evenly work is distributed

    assert len(ref_l) == len(test_l)
    shared_order = ref_l == test_l

    ref_l.sort()
    test_l.sort()
    if ref_l != test_l:
        raise RuntimeError(
            "both mechanisms for finding neighbors should iterate over all "
            "pairs of neighbors"
        )

    if ny == nz == 2:
        # this is the only case where ref_l and test_l use different mechanisms
        # for determining neighbors. We generally expect them to have different
        # orders (except maybe for weird values of nx)
        if shared_order:
            raise RuntimeError(
                f"nx = {nx}, ny = {ny}, nz = {nz} unexpectedly produced pairs "
                "of neighbors in the same order"
            )
    elif not shared_order:
        raise RuntimeError(
            f"nx = {nx}, ny = {ny}, nz = {nz} unexpectedly produced pairs "
            "of neighbors with different orders"
        )

if __name__ == '__main__':
    for nx in range(1,12):
        for ny in range(1,8):
            for nz in range(1, 8):
                test_all_neighbors(nx,ny,nz)
    
