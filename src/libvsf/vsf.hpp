#ifndef VSF_H
#define VSF_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

struct PointProps {
  // ith component of jth point (for positions and values) is located at
  // an index of `j + i*spatial_dim_stride`
  const double* positions;
  const double* values;
  const double* weights;
  size_t n_points;
  size_t n_spatial_dims;
  size_t spatial_dim_stride;
};

struct BinSpecification {
  const double* bin_edges;
  size_t n_bins;
};

struct ParallelSpec {
  size_t nproc;  // a value of 0 should probably fall back to OMP_NUM_THREADS
  bool force_sequential;  // when true, only 1 process is used, but it should
                          // partition the problem as though there were nproc
};

/// This is used to specify the statistics that will be computed.
struct StatListItem {
  /// The name of the statistic to compute.
  const char* statistic;

  /// Pointer to a struct that is designed to be passed to the constructor of
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
/// @param[in]  pairwise_op String specifying the kind of calculation: either
///     "sf" or "correlate".
///       * In the former case, we assume that the values associated with the
///         points correspond to vectors of the same dimensionality as the
///         positions (so that we can compute the velocity structure function)
///       * In the latter case, we assume that the values associated with the
///         points correspond to scalars (so that we can compute the 2pcf)
/// @param[in]  accumhandle Pointer to an accumulator handle.
/// @param[in]  bin_edges An array of monotonically increasing bin edges for
///     binning positions. This must have ``nbins + 1`` entries. The ith bin
///     includes the interval ``bin_edges[i] <= x < bin_edges[i]``.
/// @param[in]  nbins The number of position bins
/// @param[in]  parallel_spec Specifies the parallelism arguments.
///
/// @returns This returns ``true`` on success and ``false`` on failure.
bool calc_vsf_props(const PointProps points_a, const PointProps points_b,
                    const char* pairwise_op, void* accumhandle,
                    const double* bin_edges, size_t nbins,
                    const ParallelSpec parallel_spec);

/// returns whether the library was compiled with openmp support
///
/// @note
/// if we ever support more kinds of backends, we may want to revisit this
/// function (we may want to use a more generic function)
bool compiled_with_openmp(void);

#ifdef __cplusplus
}
#endif

#endif /* VSF_H */
