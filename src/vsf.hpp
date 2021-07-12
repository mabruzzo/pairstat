/// Computes properties related to the velocity structure function computed
/// between two sets of points.
///
/// @param[in]  pos_a,vel_a Arrays of ``ndim`` pointers to arrays. The inner
///     arrays should each hold ``len_a`` entries. These arrays detail
///     properties of the first set of points.
/// @param[in]  len_a The number of points in group a.
/// @param[in]  pos_b,vel_b Arrays of ``ndim`` pointers to arrays. The inner
///     arrays should each hold ``len_b`` entries. These arrays detail
///     properties of the second set of points.
/// @param[in]  len_b The number of points in group b.
/// @param[in]  The number of dimensions to consider for positions and
///     velocities.
/// @param[in]  bin_edges An array of monotonically increasing bin edges for
///     binning positions. This must have ``nbins + 1`` entries. The ith bin
///     includes the interval ``bin_edges[i] <= x < bin_edges[i]``.
/// @param[in]  nbins The number of position bins
/// @param[out] out_vals Preallocated arrays to hold the output values. In the
///     future, this may have different requirements depending on the output
///     types. Right now it's always used to just hold the mean and variance.
///     The mean is stored in the first ``nbins` entries while the mean is
///     stored in the next ``nbins`` entries.
/// @param[out] out_counts Preallocated array of ``nbins`` entries that are
///     used to store the number of pairs of points in each bin.
///
/// @returns This returns ``true`` on success and ``false`` on failure.
bool calc_vsf_props(const double **pos_a, const double **vel_a,
                    std::size_t len_a,
                    const double **pos_b, const double **vel_b,
                    std::size_t len_b,
                    uint8_t ndim,
                    const double *bin_edges, std::size_t nbins,
                    double *out_vals, int64_t *out_counts);
