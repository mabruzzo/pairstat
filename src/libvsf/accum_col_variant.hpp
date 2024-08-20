#ifndef ACCUMCOLVARIANT_H
#define ACCUMCOLVARIANT_H

#include <tuple>
#include <utility>  // std::in_place_type
#include <variant>

#include "accumulators.hpp"
#include "compound_accumulator.hpp"
#include "vsf.hpp"  // declaration of StatListItem

using HistVarCompoundAccumCollection = CompoundAccumCollection<
    std::tuple<HistogramAccumCollection, ScalarAccumCollection<VarAccum>>>;

template <class T0, class T1>
using FusedAccumCol = CompoundAccumCollection<std::tuple<T0, T1>>;

using AccumColVariant = std::variant<
    ScalarAccumCollection<MeanAccum>, ScalarAccumCollection<VarAccum>,
    HistogramAccumCollection, ScalarAccumCollection<WeightedMeanAccum>,
    WeightedHistogramAccumCollection,
    // here we start listing the fused options
    FusedAccumCol<HistogramAccumCollection, ScalarAccumCollection<MeanAccum>>,
    FusedAccumCol<HistogramAccumCollection, ScalarAccumCollection<VarAccum>>,
    FusedAccumCol<WeightedHistogramAccumCollection,
                  ScalarAccumCollection<WeightedMeanAccum>>>;

struct BuildContext_ {
  const StatListItem* stat_list;
  std::size_t stat_list_len;
  std::size_t num_dist_bins;

  template <typename T>
  AccumColVariant build1() {
    if (stat_list_len != 1) error("stat_list_len expected to be 1");

    void* arg_ptr = stat_list[0].arg_ptr;
    return AccumColVariant(std::in_place_type<T>, num_dist_bins, arg_ptr);
  }

  template <typename T0, typename T1>
  AccumColVariant build2() {
    if (stat_list_len != 2) error("stat_list_len expected to be 2");

    void* arg_ptr_0 = stat_list[0].arg_ptr;
    void* arg_ptr_1 = stat_list[1].arg_ptr;

    using MyTuple = std::tuple<T0, T1>;

    MyTuple temp_tuple = std::make_tuple(T0(num_dist_bins, arg_ptr_0),
                                         T1(num_dist_bins, arg_ptr_1));
    return AccumColVariant(std::in_place_type<CompoundAccumCollection<MyTuple>>,
                           std::move(temp_tuple));
  }
};

/// Construct an instance of AccumColVariant
inline AccumColVariant build_accum_collection(
    const StatListItem* stat_list, std::size_t stat_list_len,
    std::size_t num_dist_bins) noexcept {
  BuildContext_ ctx{stat_list, stat_list_len, num_dist_bins};
  if (stat_list_len == 0) {
    error("stat_list_len must not be 0");

  } else if (stat_list_len == 1) {
    std::string stat(stat_list[0].statistic);

    if (stat == "mean") {
      return ctx.build1<ScalarAccumCollection<MeanAccum>>();
    } else if (stat == "variance") {
      return ctx.build1<ScalarAccumCollection<VarAccum>>();
    } else if (stat == "histogram") {
      return ctx.build1<HistogramAccumCollection>();
    } else if (stat == "weightedmean") {
      return ctx.build1<ScalarAccumCollection<WeightedMeanAccum>>();
    } else if (stat == "weightedhistogram") {
      return ctx.build1<WeightedHistogramAccumCollection>();
    } else {
      error("unrecognized statistic.");
    }

  } else if (stat_list_len == 2) {
    std::string stat0(stat_list[0].statistic);
    std::string stat1(stat_list[1].statistic);

    if ((stat0 == "histogram") && (stat1 == "mean")) {
      return ctx
          .build2<HistogramAccumCollection, ScalarAccumCollection<MeanAccum>>();

    } else if ((stat0 == "histogram") && (stat1 == "variance")) {
      return ctx
          .build2<HistogramAccumCollection, ScalarAccumCollection<VarAccum>>();

    } else if ((stat0 == "weightedhistogram") && (stat1 == "weightedmean")) {
      return ctx.build2<WeightedHistogramAccumCollection,
                        ScalarAccumCollection<WeightedMeanAccum>>();

    } else {
      std::string err_msg = ("unrecognized stat combination: \"" + stat0 +
                             "\", \"" + stat1 + "\"");
      error(err_msg.c_str());
    }

  } else {
    error("stat_list_len must be 1 or 2");
  }
}

#endif /* ACCUMCOLVARIANT_H */
