import numpy as np
from scipy.spatial.distance import pdist, cdist

import pyvsf


def _twopoint_correlation_python(pos_a, pos_b, val_a, val_b, dist_bin_edges):
    if pos_b is None and val_b is None:
        distances = pdist(pos_a.T, "euclidean")
        n_points = pos_a.T.shape[0]
        products = np.empty(shape=distances.shape, dtype="f8")
        for i in range(n_points):
            for j in range(i + 1, n_points):
                # compute index according to
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
                ind = n_points * i + j - ((i + 2) * (i + 1)) // 2
                products[ind] = val_a[i] * val_a[j]
    else:
        raise NotImplementedError("NOT SUPPORTED YET")
        # distances = cdist(pos_a.T, pos_b.T, 'euclidean')
        # products

    num_bins = dist_bin_edges.size - 1
    bin_indices = np.digitize(x=distances, bins=dist_bin_edges)

    val_dict = {
        "mean": np.empty((num_bins,), dtype=np.float64),
        "counts": np.empty((num_bins,), dtype=np.int64),
    }
    for i in range(num_bins):
        # we need to add 1 to the i when checking for bin indices because
        # np.digitize assigns indices of 0 to values that fall to the left
        # of the first bin
        selected_products = products[(bin_indices == (i + 1))]
        if selected_products.size == 0:
            val_dict["mean"][i] = np.nan
        else:
            val_dict["mean"][i] = np.mean(selected_products)
        val_dict["counts"][i] = selected_products.size

    return [val_dict]


def test_twopoint_correlation():
    val_bin_edges = np.array(
        [0] + np.geomspace(start=1e-16, stop=100, num=100).tolist()
    )

    if True:
        x_a, val_a = (np.arange(6.0, 24.0).reshape(3, 6), np.arange(-9.0, 9.0, 3.0))
        bin_edges = np.array([0.0, 5.0, 10.0])

    ref = _twopoint_correlation_python(x_a, None, val_a, None, bin_edges)
    actual = pyvsf.twopoint_correlation(x_a, None, val_a, None, bin_edges)
    for key in ["mean", "counts"]:
        assert np.all(ref[0][key] == actual[0][key])


if __name__ == "__main__":
    test_twopoint_correlation()
