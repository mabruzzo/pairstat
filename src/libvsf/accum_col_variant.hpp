#ifndef ACCUMCOLVARIANT_H
#define ACCUMCOLVARIANT_H

#include <tuple>
#include <utility>  // std::in_place_type
#include <variant>

#include "accumulators.hpp"
#include "compound_accumulator.hpp"
#include "vsf.hpp"  // declaration of StatListItem

using HistVarianceTuple =
    std::tuple<HistogramAccumCollection, ScalarAccumCollection<VarAccum>>;
using HistVarCompoundAccumCollection =
    CompoundAccumCollection<HistVarianceTuple>;

using AccumColVariant =
    std::variant<ScalarAccumCollection<MeanAccum>,
                 ScalarAccumCollection<VarAccum>, HistogramAccumCollection,
                 HistVarCompoundAccumCollection,
                 ScalarAccumCollection<WeightedMeanAccum>>;

/// Construct an instance of AccumColVariant
inline AccumColVariant build_accum_collection(
    const StatListItem* stat_list, std::size_t stat_list_len,
    std::size_t num_dist_bins) noexcept {
  if (stat_list_len == 0) {
    error("stat_list_len must not be 0");
  } else if (stat_list_len == 1) {
    std::string stat_str(stat_list[0].statistic);
    void* accum_arg_ptr = stat_list[0].arg_ptr;

    if (stat_str == "mean") {
      return AccumColVariant(
          std::in_place_type<ScalarAccumCollection<MeanAccum>>, num_dist_bins,
          accum_arg_ptr);

    } else if (stat_str == "variance") {
      return AccumColVariant(
          std::in_place_type<ScalarAccumCollection<VarAccum>>, num_dist_bins,
          accum_arg_ptr);

    } else if (stat_str == "histogram") {
      return AccumColVariant(std::in_place_type<HistogramAccumCollection>,
                             num_dist_bins, accum_arg_ptr);

    } else {
      error("unrecognized statistic.");
    }
  } else if (stat_list_len == 2) {
    // Note: we might be able to do something clever where we call this
    // function to construct each of the individual accumulators

    std::string stat_str_a(stat_list[0].statistic);
    void* accum_arg_ptr_a = stat_list[0].arg_ptr;

    std::string stat_str_b(stat_list[1].statistic);
    void* accum_arg_ptr_b = stat_list[1].arg_ptr;

    if ((stat_str_a == "histogram") && (stat_str_b == "variance")) {
      HistVarianceTuple temp_tuple = std::make_tuple(
          HistogramAccumCollection(num_dist_bins, accum_arg_ptr_a),
          ScalarAccumCollection<VarAccum>(num_dist_bins, accum_arg_ptr_b));
      return AccumColVariant(std::in_place_type<HistVarCompoundAccumCollection>,
                             std::move(temp_tuple));

    } else {
      error("unrecognized stat combination.");
    }

  } else {
    error("stat_list_len must be 1 or 2");
  }
}

#endif /* ACCUMCOLVARIANT_H */
