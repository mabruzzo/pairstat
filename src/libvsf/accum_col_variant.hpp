#ifndef ACCUMCOLVARIANT_H
#define ACCUMCOLVARIANT_H

#include <tuple>
#include <utility>  // std::in_place_type
#include <variant>

#include "accumulators.hpp"
#include "fused_accumulator.hpp"
#include "vsf.hpp"  // declaration of StatListItem

template <class T0, class T1>
using MyFusedAccumCol = FusedAccumCollection<std::tuple<T0, T1>>;

template <int Order>
using MomentAccum =
    Accumulator<OriginMomentStatistic<Order, std::int64_t, false>>;

template <int Order>
using MomentWithVarAccum =
    Accumulator<OriginMomentStatistic<Order, std::int64_t, true>>;

template <int Order>
using WeightedMomentAccum =
    Accumulator<OriginMomentStatistic<Order, double, false>>;

template <int Order>
using WeightedMomentWithVarAccum =
    Accumulator<OriginMomentStatistic<Order, double, true>>;

using AccumColVariant = std::variant<
    MomentAccum<1>, WeightedMomentAccum<1>, MomentWithVarAccum<1>,
    WeightedMomentWithVarAccum<1>, MomentAccum<2>, WeightedMomentAccum<2>,
    MomentWithVarAccum<2>, WeightedMomentWithVarAccum<2>, MomentAccum<3>,
    WeightedMomentAccum<3>, MomentWithVarAccum<3>,
    WeightedMomentWithVarAccum<3>, MomentAccum<4>, WeightedMomentAccum<4>,
    MomentWithVarAccum<4>, WeightedMomentWithVarAccum<4>,
    // list Histogram accumulators
    HistogramAccumCollection, WeightedHistogramAccumCollection,
    // here we start listing the fused options
    MyFusedAccumCol<HistogramAccumCollection, MomentAccum<1>>,
    MyFusedAccumCol<HistogramAccumCollection, MomentWithVarAccum<1>>,
    MyFusedAccumCol<WeightedHistogramAccumCollection, WeightedMomentAccum<1>>,
    MyFusedAccumCol<WeightedHistogramAccumCollection,
                    WeightedMomentWithVarAccum<1>>>;

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
    return AccumColVariant(std::in_place_type<FusedAccumCollection<MyTuple>>,
                           std::move(temp_tuple));
  }
};

inline std::string get_stat_name_(const StatListItem& item) {
  std::string name(item.statistic);
  // remap some older names
  if (name == "mean") return "omoment1";
  if (name == "weightedmean") return "weightedomoment1";
  if (name == "variance") return "omoment1_var";
  if (name == "weightedvariance") return "weightedomoment1_var";
  return name;
}

/// Construct an instance of AccumColVariant
inline AccumColVariant build_accum_collection(
    const StatListItem* stat_list, std::size_t stat_list_len,
    std::size_t num_dist_bins) noexcept {
  BuildContext_ ctx{stat_list, stat_list_len, num_dist_bins};
  if (stat_list_len == 0) {
    error("stat_list_len must not be 0");

  } else if (stat_list_len == 1) {
    std::string stat = get_stat_name_(stat_list[0]);

    // this could be improved a lot! We could write a simple regex for most of
    // these cases and then dispatch to a factory template function
    if (stat == "omoment1") {
      return ctx.build1<MomentAccum<1>>();
    } else if (stat == "weightedomoment1") {
      return ctx.build1<WeightedMomentAccum<1>>();
    } else if (stat == "omoment1_var") {
      return ctx.build1<MomentWithVarAccum<1>>();
    } else if (stat == "weightedomoment1_var") {
      return ctx.build1<WeightedMomentWithVarAccum<1>>();
    } else if (stat == "omoment2") {
      return ctx.build1<MomentAccum<2>>();
    } else if (stat == "weightedomoment2") {
      return ctx.build1<WeightedMomentAccum<2>>();
    } else if (stat == "omoment2_var") {
      return ctx.build1<MomentWithVarAccum<2>>();
    } else if (stat == "weightedomoment2_var") {
      return ctx.build1<WeightedMomentWithVarAccum<2>>();
    } else if (stat == "omoment3") {
      return ctx.build1<MomentAccum<3>>();
    } else if (stat == "weightedomoment3") {
      return ctx.build1<WeightedMomentAccum<3>>();
    } else if (stat == "omoment3_var") {
      return ctx.build1<MomentWithVarAccum<3>>();
    } else if (stat == "weightedomoment3_var") {
      return ctx.build1<WeightedMomentWithVarAccum<3>>();
    } else if (stat == "omoment4") {
      return ctx.build1<MomentAccum<4>>();
    } else if (stat == "weightedomoment4") {
      return ctx.build1<WeightedMomentAccum<4>>();
    } else if (stat == "omoment4_var") {
      return ctx.build1<MomentWithVarAccum<4>>();
    } else if (stat == "weightedomoment4_var") {
      return ctx.build1<WeightedMomentWithVarAccum<4>>();
    } else if (stat == "histogram") {
      return ctx.build1<HistogramAccumCollection>();
    } else if (stat == "weightedvariance") {
      return ctx.build1<WeightedMomentWithVarAccum<1>>();
    } else if (stat == "weightedhistogram") {
      return ctx.build1<WeightedHistogramAccumCollection>();
    } else {
      error("unrecognized statistic.");
    }

  } else if (stat_list_len == 2) {
    // at this point, I'm skeptical that the FusedAccumulator was a good idea
    // -> in the future, we will remove it (or at the very least, we'll try to
    //    roughly benchmark its speed)

    std::string stat0 = get_stat_name_(stat_list[0]);
    std::string stat1 = get_stat_name_(stat_list[1]);

    if ((stat0 == "histogram") && (stat1 == "omoment1")) {
      return ctx.build2<HistogramAccumCollection, MomentAccum<1>>();

    } else if ((stat0 == "histogram") && (stat1 == "omoment1_var")) {
      return ctx.build2<HistogramAccumCollection, MomentWithVarAccum<1>>();

    } else if ((stat0 == "weightedhistogram") &&
               (stat1 == "weightedomoment1")) {
      return ctx
          .build2<WeightedHistogramAccumCollection, WeightedMomentAccum<1>>();

    } else if ((stat0 == "weightedhistogram") &&
               (stat1 == "weightedomoment1_var")) {
      return ctx
          .build2<WeightedHistogramAccumCollection, WeightedMomentAccum<1>>();

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
