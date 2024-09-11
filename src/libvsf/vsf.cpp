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

enum class PairOperation { vec_diff, correlate };

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

/// returns the number of dimensions used in a single
constexpr int val_vec_rank_(PairOperation op) {
  return op == PairOperation::correlate ? 1 : 3;
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
      if constexpr (choice == PairOperation::correlate) {
        // compute the product of 2 scalars
        op_rslt = val_a.vals[0] * val_b.vals[0];
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
void calc_vsf_props_helper_(const PointProps points_a,
                            const PointProps points_b,
                            const double* dist_sqr_bin_edges, std::size_t nbins,
                            AccumCollection& accumulators,
                            bool duplicated_points) {
  if (duplicated_points) {
    process_data<AccumCollection, true, choice>(
        points_a, points_b, dist_sqr_bin_edges, nbins, accumulators);
  } else {
    process_data<AccumCollection, false, choice>(
        points_a, points_b, dist_sqr_bin_edges, nbins, accumulators);
  }
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
    // in the current implementation, this doesn't really matter

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

// we annotate the following with the maybe_unused attribute to suppress
// warnings when compiling without openmp
[[maybe_unused]] std::size_t get_nominal_nproc_(
    const ParallelSpec& parallel_spec) noexcept {
  if (parallel_spec.nproc == 0) {
    // this approach is crude. OMP_NUM_THREADS doesn't need to be an int
    char* var_val = std::getenv("OMP_NUM_THREADS");
    if (var_val == nullptr) {
      return 1;
    } else {
      int tmp = std::atoi(var_val);
      if (tmp <= 0) {
        error("OMP_NUM_THREADS has an invalid value");
      } else {
        return tmp;
      }
    }
  } else {
    return parallel_spec.nproc;
  }
}

template <typename AccumCollection, PairOperation choice>
void calc_vsf_props_parallel_(const PointProps points_a,
                              const PointProps points_b,
                              const double* dist_sqr_bin_edges,
                              std::size_t nbins,
                              const ParallelSpec parallel_spec,
                              AccumCollection& accumulators,
                              bool duplicated_points) noexcept {
#ifndef _OPENMP
  error(
      "calc_vsf_props_parallel_ should not be called unless the library "
      "is compiled with support for OPENMP");
#else
  std::size_t nominal_nproc = get_nominal_nproc_(parallel_spec);

  const TaskItFactory factory(nominal_nproc, points_a.n_points,
                              (duplicated_points) ? 0 : points_b.n_points);

  // this may be less than the value from parallel_spec.nproc
  const std::size_t nproc = factory.effective_nproc();

  omp_set_num_threads(nproc);
  omp_set_dynamic(0);

  // initialize vector where the accumulator collection that is used to
  // process each partition will be stored.
  // (This assumes that accumulators hasn't been used yet - we just clone it)
  std::vector<AccumCollection> partition_dest;
  partition_dest.reserve(nproc);
  for (std::size_t i = 0; i < nproc; i++) {
    AccumCollection copy = accumulators;
    partition_dest.push_back(copy);
  }

  const bool use_parallel = ((!parallel_spec.force_sequential) && (nproc > 1));

// printf("About to enter parallel region.\n"
//        "  use_parallel: %d, nproc = %zu, n_partitions = %zu\n",
//        (int)use_parallel, nproc, (std::size_t)factory.n_partitions());

// now actually compute the number of statistics
#pragma omp parallel if (use_parallel)
  {
// the proc_id value probably won't align with the actual process id
#pragma omp for schedule(static, 1)
    for (std::size_t proc_id = 0; proc_id < nproc; proc_id++) {
      // make a local copy. Do this so that the heap allocation corresponds
      // to a location that is fast for the current process to access.
      AccumCollection local_accums(partition_dest[proc_id]);

      process_TaskIt_<AccumCollection, choice>(
          points_a, points_b, dist_sqr_bin_edges, nbins, local_accums,
          duplicated_points, factory.build_TaskIt(proc_id));

      partition_dest[proc_id] = local_accums;
    }

#pragma omp barrier  // I think the barrier may be implied
  }

  // lastly, let's consolidate the values
  accumulators = partition_dest[0];
  for (std::size_t i = 1; i < nproc; i++) {
    accumulators.consolidate_with_other(partition_dest[i]);
  }
#endif
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
  PairOperation operation_choice = PairOperation::vec_diff;
  if (tmp == "correlate") {
    if (requires_weight) error("correlation incompatible with weighted stat");
    operation_choice = PairOperation::correlate;
  } else if (tmp == "sf") {
    operation_choice = PairOperation::vec_diff;
  } else {
    return false;
  }

  if (requires_weight && (points_a.weights == nullptr)) {
    return false;
  }

#ifdef _OPENMP
  const bool use_serial = parallel_spec.nproc == 1;
#else
  const bool use_serial = true;
#endif

  // now actually use the accumulators to compute that statistics
  auto func = [=](auto& accumulators) {
    using AccumCollection = std::decay_t<decltype(accumulators)>;
    if (use_serial) {
      if (operation_choice == PairOperation::vec_diff) {
        calc_vsf_props_helper_<AccumCollection, PairOperation::vec_diff>(
            points_a, my_points_b, dist_sqr_bin_edges_vec.data(), nbins,
            accumulators, duplicated_points);
      } else if (operation_choice == PairOperation::correlate) {
        calc_vsf_props_helper_<AccumCollection, PairOperation::correlate>(
            points_a, my_points_b, dist_sqr_bin_edges_vec.data(), nbins,
            accumulators, duplicated_points);
      }

    } else {
      if (operation_choice == PairOperation::vec_diff) {
        calc_vsf_props_parallel_<AccumCollection, PairOperation::vec_diff>(
            points_a, my_points_b, dist_sqr_bin_edges_vec.data(), nbins,
            parallel_spec, accumulators, duplicated_points);
      } else if (operation_choice == PairOperation::correlate) {
        calc_vsf_props_parallel_<AccumCollection, PairOperation::correlate>(
            points_a, my_points_b, dist_sqr_bin_edges_vec.data(), nbins,
            parallel_spec, accumulators, duplicated_points);
      }
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
