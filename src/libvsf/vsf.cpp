#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>  // std::getenv
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "accum_col_variant.hpp"
#include "partition.hpp"
#include "utils.hpp"
#include "vsf.hpp"

/// @def OMP_PRAGMA
/// Macro used to wrap OpenMP's pragma directives.
///
/// When the program:
///  * is compiled with OpenMP, the pragma contents are honored.
///  * is NOT compiled with OpenMP, the pragma contents are ignored.
///
/// @note
/// This macro is implemented using the ``_Pragma`` operator, described
/// [here](https://en.cppreference.com/w/cpp/preprocessor/impl). More details
/// can be found [here](https://gcc.gnu.org/onlinedocs/cpp/Pragmas.html).
#ifdef _OPENMP
#define OMP_PRAGMA(x) _Pragma(#x)
#else
#define OMP_PRAGMA(x) /* ... */
#endif

enum class PairOperation { vector_diff, scalar_correlate, vector_correlate };

// the anonymous namespace informs the compiler that the contents are only used
// in the local compilation unit (facillitating more optimizations)
namespace {  // anonymous namespace

template <bool cond>
FORCE_INLINE double conditional_get(const double* ptr, std::size_t i) {
  if constexpr (cond) {
    return ptr[i];
  } else {
    return 0.0;
  }
}

template <int D>
struct MathVec {
  static_assert(D >= 1 && D <= 4, "D must be 1, 2, or 3");
  const double vals[D];
};

template <int D>
static FORCE_INLINE MathVec<D> load_vec_(const double* ptr, std::size_t index,
                                         std::size_t stride) {
  if constexpr (D == 1) {
    return {{ptr[index]}};
  } else if constexpr (D == 2) {
    return {{ptr[index], ptr[index + stride]}};
  } else {
    return {{ptr[index], ptr[index + stride], ptr[index + 2 * stride]}};
  }
}

/// returns the number of dimensions used in a single value
constexpr int val_vec_rank_(PairOperation op) {
  return op == PairOperation::scalar_correlate ? 1 : 3;
}

template <PairOperation op>
inline constexpr int ValueRank =
    std::integral_constant<int, val_vec_rank_(op)>::value;

FORCE_INLINE double calc_dist_sqr(MathVec<3> pos_a, MathVec<3> pos_b) noexcept {
  double dx = pos_a.vals[0] - pos_b.vals[0];
  double dy = pos_a.vals[1] - pos_b.vals[1];
  double dz = pos_a.vals[2] - pos_b.vals[2];
  return dx * dx + dy * dy + dz * dz;
}

// this could be refactored into something a lot more elegant!
template <class AccumCollection, bool duplicated_points, PairOperation choice>
void process_data(const PointProps points_a, const PointProps points_b,
                  const double* dist_sqr_bin_edges, std::size_t nbins,
                  AccumCollection& accumulators) {
  // define a compile-time constant to determine whether we should use weights
  using use_weights = std::bool_constant<AccumCollection::requires_weight>;

  // define a ValueRank
  using ValueType = MathVec<ValueRank<choice>>;
  // this assumes 3D positions

  const std::size_t n_points_a = points_a.n_points;
  const std::size_t spatial_dim_stride_a = points_a.spatial_dim_stride;
  const double* pos_a = points_a.positions;
  const double* values_a = points_a.values;
  const double* weights_a = points_a.weights;

  const std::size_t n_points_b = points_b.n_points;
  const std::size_t spatial_dim_stride_b = points_b.spatial_dim_stride;
  const double* pos_b = points_b.positions;
  const double* values_b = points_b.values;
  const double* weights_b = points_b.weights;

  for (std::size_t i_a = 0; i_a < n_points_a; i_a++) {
    // When duplicated_points is true, points_a is the same as points_b. In
    // that case, take some care to avoid duplicating pairs
    std::size_t i_b_start = (duplicated_points) ? i_a + 1 : 0;

    // load current position, value and (if applicable) weight from points_a
    const MathVec<3> loc_a = load_vec_<3>(pos_a, i_a, spatial_dim_stride_a);
    const ValueType val_a =
        load_vec_<ValueRank<choice>>(values_a, i_a, spatial_dim_stride_a);

    const double weight_a = conditional_get<use_weights::value>(weights_a, i_a);

    for (std::size_t i_b = i_b_start; i_b < n_points_b; i_b++) {
      const MathVec<3> loc_b = load_vec_<3>(pos_b, i_b, spatial_dim_stride_b);
      const ValueType val_b =
          load_vec_<ValueRank<choice>>(values_b, i_b, spatial_dim_stride_b);

      // compute the squared distance between loc_a and loc_b
      double dist_sqr = calc_dist_sqr(loc_a, loc_b);

      double op_rslt;
      if constexpr (choice == PairOperation::scalar_correlate) {
        // compute the product of 2 scalars
        op_rslt = val_a.vals[0] * val_b.vals[0];

      } else if constexpr (choice == PairOperation::vector_correlate) {
        // compute the dot product of 2 vectors
        op_rslt = ((val_a.vals[0] * val_b.vals[0]) +
                   ((val_a.vals[1] * val_b.vals[1]) +
                    (val_a.vals[2] * val_b.vals[2])));

      } else {
        // compute the magnitude of the difference between 2 vectors
        op_rslt = std::sqrt(calc_dist_sqr(val_a, val_b));
      }

      // determine the bin that we lie within
      std::size_t bin_ind =
          identify_bin_index(dist_sqr, dist_sqr_bin_edges, nbins);

      if (bin_ind < nbins) {
        if constexpr (use_weights::value) {
          double product = weight_a * weights_b[i_b];
          accumulators.add_entry(bin_ind, op_rslt, product);
        } else {
          accumulators.add_entry(bin_ind, op_rslt);
        }
      }
    } /* looping through points_b */
  } /* looping through points_a */
}

template <typename AccumCollection, PairOperation choice>
void process_TaskIt_(const PointProps points_a, const PointProps points_b,
                     const double* dist_sqr_bin_edges, std::size_t nbins,
                     AccumCollection& accumulators, bool duplicated_points,
                     TaskIt task_iter) noexcept {
  const bool ignore_weights = points_a.weights == nullptr;
  while (task_iter.has_next()) {
    const StatTask stat_task = task_iter.next();

    // keep in mind that depending on choice, the PointProps::values can
    // represent ether:
    //   -> a list of 3D vectors OR
    //   -> a list of scalar values
    // with the current assumptions about indexing order, this doesn't really
    // matter (but it will matter if we ever revisit these assumptions)

    const PointProps cur_points_a = {
        points_a.positions + stat_task.start_A,
        points_a.values + stat_task.start_A,
        ignore_weights ? nullptr : points_a.weights + stat_task.start_A,
        stat_task.stop_A - stat_task.start_A,  // = n_points
        points_a.n_spatial_dims,
        points_a.spatial_dim_stride};

    const PointProps cur_points_b = {
        points_b.positions + stat_task.start_B,
        points_b.values + stat_task.start_B,
        ignore_weights ? nullptr : points_b.weights + stat_task.start_B,
        stat_task.stop_B - stat_task.start_B,  // = n_points
        points_b.n_spatial_dims,
        points_b.spatial_dim_stride};

    if (duplicated_points) {
      if ((stat_task.start_B == stat_task.stop_B) & (stat_task.stop_B == 0)) {
        // not a typo, use cur_points_a twice
        process_data<AccumCollection, true, choice>(cur_points_a, cur_points_a,
                                                    dist_sqr_bin_edges, nbins,
                                                    accumulators);
      } else {
        process_data<AccumCollection, false, choice>(cur_points_a, cur_points_b,
                                                     dist_sqr_bin_edges, nbins,
                                                     accumulators);
      }
    } else {
      process_data<AccumCollection, false, choice>(
          cur_points_a, cur_points_b, dist_sqr_bin_edges, nbins, accumulators);
    }
  }
}

std::size_t get_nominal_nproc_(const ParallelSpec& parallel_spec) noexcept {
#ifndef _OPENMP
  return (parallel_spec.nproc == 0) ? 1 : parallel_spec.nproc;
#else
  if (parallel_spec.nproc == 0) {
    // this approach is crude. OMP_NUM_THREADS may not be an int
    char* var_val = std::getenv("OMP_NUM_THREADS");
    if (var_val == nullptr) return 1;
    int tmp = std::atoi(var_val);
    if (tmp <= 0) error("OMP_NUM_THREADS has an invalid value");
    return tmp;
  }
  return parallel_spec.nproc;
#endif
}

template <typename AccumCollection, PairOperation choice>
void calc_pair_props_(const PointProps points_a, const PointProps points_b,
                      const double* dist_sqr_bin_edges, std::size_t nbins,
                      const ParallelSpec parallel_spec,
                      AccumCollection& accumulators,
                      bool duplicated_points) noexcept {
  std::size_t nominal_nproc = get_nominal_nproc_(parallel_spec);

  /// construct TaskItFactory, which encapsulates the logic for partitioning
  /// out the calculation into groups of tasks.
  const TaskItFactory factory(nominal_nproc, points_a.n_points,
                              (duplicated_points) ? 0 : points_b.n_points);

  if (nominal_nproc == 1) {
    if ((factory.n_partitions() != 1) || (factory.effective_nproc() != 1)) {
      error(
          "A bug appeared to have occured. When there is 1 process, the "
          "calculations should only be divided into 1 part");
    }
    process_TaskIt_<AccumCollection, choice>(
        points_a, points_b, dist_sqr_bin_edges, nbins, accumulators,
        duplicated_points, factory.build_TaskIt(0));
    return;
  }

  // get the number of task-groups.
  // - each task_group is a group of 1 or more tasks
  // - each thread must operates on a complete task_group (thus, it doesn't
  //   make sense to have more threads than task_sets)
  // - This number may be less than the nominal number of available processes
  //   (i.e. signaled by parallel_spec.nproc). In this scenario, there may have
  //   been an awkward amount of work to partition among the available
  //   processes (i.e. we got lazy and did something simple). Alternatively,
  //   we may have decided that there wasn't enough work to warrant spreading
  //   it across the max number of threads processes.
  const std::size_t ntask_groups = factory.effective_nproc();

#ifdef _OPENMP
  const bool use_parallel =
      ((!parallel_spec.force_sequential) && (ntask_groups > 1));

  omp_set_num_threads(ntask_groups);
  omp_set_dynamic(0);
#else
  const bool use_parallel = false;
#endif

  // initialize vector where the accumulator collection that is used to
  // process each partition will be stored.
  // (This assumes that accumulators hasn't been used yet - we just clone it)
  std::vector<AccumCollection> partition_dest;
  partition_dest.reserve(ntask_groups);
  for (std::size_t i = 0; i < ntask_groups; i++) {
    AccumCollection copy = accumulators;
    partition_dest.push_back(copy);
  }

  // now actually compute the number of statistics

  // this first pragma, conditionally specifies the start of a parallel region
  // -> when ``use_parallel == true``, then openmp distributes the work of the
  //    enclosed for-loop among the threads
  // -> when ``use_parallel == false``, then a single thread executes the inner
  //    for-loop all by itself
  // the bookkeeping with partition_dest should ensure that the answer is
  // bitwise identical, regardless of the action taken by use_parrallel

  OMP_PRAGMA(omp parallel if (use_parallel)) {
    // NOTE: even when use_parallel==true, part_id probably won't match the
    //       value provided by omp_get_thread_num

    // clang-format off
    OMP_PRAGMA(omp for schedule(static, 1))
    for (std::size_t group_id = 0; group_id < ntask_groups; group_id++) {
      // clang-format on

      // make a local copy. Do this so that the heap allocation corresponds
      // to a location that is fast for the current process to access.
      AccumCollection local_accums(partition_dest[group_id]);

      process_TaskIt_<AccumCollection, choice>(
          points_a, points_b, dist_sqr_bin_edges, nbins, local_accums,
          duplicated_points, factory.build_TaskIt(group_id));

      partition_dest[group_id] = local_accums;
    }

    OMP_PRAGMA(omp barrier)  // I think the barrier may be implied
  }

  // lastly, let's consolidate the values
  accumulators = partition_dest[0];
  for (std::size_t group_id = 1; group_id < ntask_groups; group_id++) {
    accumulators.consolidate_with_other(partition_dest[group_id]);
  }
}

}  // namespace

bool calc_vsf_props(const PointProps points_a, const PointProps points_b,
                    const char* pairwise_op, const StatListItem* stat_list,
                    std::size_t stat_list_len, const double* bin_edges,
                    std::size_t nbins, const ParallelSpec parallel_spec,
                    double* out_flt_vals, int64_t* out_i64_vals) {
  const bool duplicated_points =
      ((points_b.positions == nullptr) && (points_b.values == nullptr));

  const PointProps my_points_b = (duplicated_points) ? points_a : points_b;

  if (nbins == 0) {
    return false;
  } else if (points_a.n_spatial_dims != 3) {
    return false;
  } else if (my_points_b.n_spatial_dims != 3) {
    return false;
  } else if ((points_a.positions == nullptr) || (points_a.values == nullptr)) {
    return false;
  } else if ((points_a.weights == nullptr) !=
             (my_points_b.weights == nullptr)) {
    return false;
  }

  // recompute the bin edges so that they are stored as squared distances
  std::vector<double> dist_sqr_bin_edges_vec(nbins + 1);
  for (std::size_t i = 0; i < (nbins + 1); i++) {
    if (bin_edges[i] < 0) {
      // It doesn't really matter how we handle negative bin edges (since
      // distances are non-negative), as long as dist_sqr_bin_edges
      // monotonically increases.
      dist_sqr_bin_edges_vec[i] = bin_edges[i];
    } else {
      dist_sqr_bin_edges_vec[i] = bin_edges[i] * bin_edges[i];
    }
  }

  // construct accumulators (they're stored in a std::variant for convenience)
  AccumColVariant accumulators =
      build_accum_collection(stat_list, stat_list_len, nbins);

  bool requires_weight =
      std::visit([](auto& a) { return a.requires_weight; }, accumulators);

  std::string tmp(pairwise_op);
  PairOperation operation_choice = PairOperation::vector_diff;
  if (tmp == "scalar_correlate") {
    operation_choice = PairOperation::scalar_correlate;
  } else if (tmp == "vector_correlate") {
    operation_choice = PairOperation::vector_correlate;
  } else if (tmp == "sf") {
    operation_choice = PairOperation::vector_diff;
  } else {
    return false;
  }

  if (requires_weight && (points_a.weights == nullptr)) {
    return false;
  }

  // now actually use the accumulators to compute that statistics
  auto func = [=](auto& accumulators) {
    using AccumCollection = std::decay_t<decltype(accumulators)>;

    if (operation_choice == PairOperation::vector_diff) {
      calc_pair_props_<AccumCollection, PairOperation::vector_diff>(
          points_a, my_points_b, dist_sqr_bin_edges_vec.data(), nbins,
          parallel_spec, accumulators, duplicated_points);
    } else if (operation_choice == PairOperation::scalar_correlate) {
      calc_pair_props_<AccumCollection, PairOperation::scalar_correlate>(
          points_a, my_points_b, dist_sqr_bin_edges_vec.data(), nbins,
          parallel_spec, accumulators, duplicated_points);
    } else if (operation_choice == PairOperation::vector_correlate) {
      calc_pair_props_<AccumCollection, PairOperation::vector_correlate>(
          points_a, my_points_b, dist_sqr_bin_edges_vec.data(), nbins,
          parallel_spec, accumulators, duplicated_points);
    }
  };
  std::visit(func, accumulators);

  // now copy the results from the accumulators to the output array
  std::visit([=](auto& accums) { accums.copy_vals(out_flt_vals); },
             accumulators);
  std::visit([=](auto& accums) { accums.copy_vals(out_i64_vals); },
             accumulators);

  return true;
}

bool compiled_with_openmp() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}
