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

/// This is used to specify the statistics that will be computed.
struct StatListItem{
  /// The name of the statistic to compute.
  const char* statistic;

  /// Pointer to a struct that is designed to be passed to the construtor of
  /// the accumulator for the specified statistic. In most cases, this should
  /// just be a nullptr
  void* arg_ptr;
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
/// @param[in]  stat_list Pointer to an array of 1 or more StatListItems that
///     provide details about the statistics that will be computed.
/// @param[in]  stat_list_len Specifies the number of entries in stat_list.
/// @param[in]  bin_edges An array of monotonically increasing bin edges for
///     binning positions. This must have ``nbins + 1`` entries. The ith bin
///     includes the interval ``bin_edges[i] <= x < bin_edges[i]``.
/// @param[in]  nbins The number of position bins
/// @param[out] out_flt_vals Preallocated arrays to hold the output floating
///     point values.
/// @param[out] out_i64_vals Preallocated array of ``nbins`` entries that are
///     used to store the number of pairs of points in each bin.
///
/// @returns This returns ``true`` on success and ``false`` on failure.
bool calc_vsf_props(const PointProps points_a, const PointProps points_b,
                    const StatListItem* stat_list, std::size_t stat_list_len,
                    const double *bin_edges, std::size_t nbins,
                    double *out_flt_vals, int64_t *out_i64_vals) noexcept;

#ifdef __cplusplus
}
#endif

#endif /* VSF_H */
