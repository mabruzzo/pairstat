#ifndef VSF_H
#define VSF_H


struct PointProps{
  const double * positions;
  const double * velocities;
  std::size_t n_points;
  std::size_t n_spatial_dims;
};

struct BinSpecification{
  const double * bin_edges;
  std::size_t n_bins;
};

#ifdef __cplusplus
extern "C" {
#endif

/// Computes properties related to the velocity structure function computed
/// between two sets of points.
///
/// @param[in]  points_a Struct holding first set of positions and velocities
/// @param[in]  points_b Struct holding second set of positions and velocities.
///     In the event that the positions and velocities pointers are each
///     nullptrs, then pairwise distances are just computed for points_a
///     (without duplicating any pairs).
/// @param[in]  statistic The name of the statistic to compute.
/// @param[in]  bin_edges An array of monotonically increasing bin edges for
///     binning positions. This must have ``nbins + 1`` entries. The ith bin
///     includes the interval ``bin_edges[i] <= x < bin_edges[i]``.
/// @param[in]  nbins The number of position bins
/// @param[in]  accum_arg_ptr This is a pointer to a struct that is designed to
///     be passed to the construction of some accumulators. In most cases, this
///     should be a nullptr.
/// @param[out] out_flt_vals Preallocated arrays to hold the output floating
///     point values.
/// @param[out] out_i64_vals Preallocated array of ``nbins`` entries that are
///     used to store the number of pairs of points in each bin.
///
/// @returns This returns ``true`` on success and ``false`` on failure.
bool calc_vsf_props(const PointProps points_a, const PointProps points_b,
		    const char* statistic,
                    const double *bin_edges, std::size_t nbins,
                    void* accum_arg_ptr,
                    double *out_flt_vals, int64_t *out_i64_vals) noexcept;

#ifdef __cplusplus
}
#endif

#endif /* VSF_H */
