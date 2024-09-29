#ifndef ACCUMULATORS_H
#define ACCUMULATORS_H

// Here's the basic overview of the stuff in this file:
// -> we define some "statistic classes." These are used to implement the
//    logic for computing various moments. This logic is decoupled from the
//    storage required to store intermediate results.
// -> we use a Statistic class to specialize the Accumulator class template.
//    the Accumulator<Statistic> also tracks the storage for the statistic. It
//    also implements a few other useful pieces of functionality.

#include <algorithm>  // std::fill
#include <cstdint>    // std::int64_t
#include <string>
#include <type_traits>  // std::is_same_v
#include <utility>      // std::pair
#include <vector>

#include "statdataview.hpp"
#include "utils.hpp"  // error
#include "vsf.hpp"    // BinSpecification

/// defining a construct like this is a common workaround used to raise a
/// compile-time error in the else-branch of a constexpr-if statement.
template <class>
inline constexpr bool dummy_false_v_ = false;

/// compute the total count
///
/// @note
/// There is some question about what the most numerically stable way to do
/// this actually is. Some of this debate is highlighted inside of this
/// function...
template <typename T>
inline double consolidate_mean_(double primary_mean, T primary_weight,
                                double other_mean, T other_weight,
                                double total_weight) {
#if 1
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
  // suggests that approach is more stable when the values of
  // this->count and other.count are approximately the same and large
  return (primary_weight * primary_mean + other_weight * other_mean) /
         total_weight;
#else
  // in the other limit, (other_weight is smaller and close to 1) the following
  // may be more stable
  double delta = other_mean - primary_mean;
  return primary_mean + (delta * other_weight / total_weight);
#endif
}

/// identify the index of the bin where x lies.
///
/// @param x The value that is being queried
/// @param bin_edges An array of monotonically increasing bin edges. This
///    must have ``nbins + 1`` entries. The ith bin includes the interval
///    ``bin_edges[i] <= x < bin_edges[i]``.
/// @param nbins The number of bins. This is expected to be at least 1.
///
/// @returns index The index that ``x`` belongs in. If ``x`` doesn't lie in
///    any bins, ``nbins`` is returned.
///
/// @notes
/// At the moment we are using a binary search algorithm. In the future, we
/// might want to assess the significance of branch mispredictions.
template <typename T>
std::size_t identify_bin_index(T x, const T* bin_edges, std::size_t nbins) {
  const T* bin_edges_end = bin_edges + nbins + 1;
  const T* rslt = std::lower_bound(bin_edges, bin_edges_end, x);
  // rslt is a pointer to the first value that is "not less than" x
  std::size_t index_p_1 = std::distance(bin_edges, rslt);

  if (index_p_1 == 0 || index_p_1 == (nbins + 1)) {
    return nbins;
  } else {
    return index_p_1 - 1;
  }
}

/// Describes a statistic property
struct PropDescr {
  const char* name = nullptr;
  bool is_f64 = true;
  int count = 1;
};

/// Implements a statistic (for measuring central moments) that can be
/// used to specialize Accumulator
///
/// @tparam Order Specifies the highest order moment that is computed. We also
///    compute all lower order moments
/// @tparam CountT Specifies the type of the counts. When this is int64_t
///    the class does standard stuff. When it is double, the class computes
///    the weighted moments
///
/// As a historical note, we originally had separate accumulators for mean,
/// variance, and weighted mean.
///
/// @note
/// The 1st and 2nd second moments exactly the same as mean and variance.
/// While the 3rd and 4th moments are related to skew and kurtosis, they are
/// not exactly the same.
///
/// @note
/// In the future, to generalize to higher order moments, we can look at
/// https://zenodo.org/records/1232635
template <int Order, typename CountT = int64_t>
class CentralMomentStatistic {
  static_assert((1 <= Order) && (Order <= 3),
                "at the moment we only allow 1 <= Order <=3");
  static_assert(std::is_same_v<CountT, std::int64_t> ||
                    std::is_same_v<CountT, double>,
                "invalid type was used.");
  static_assert(((std::is_same_v<CountT, double> && (Order <= 2)) ||
                 std::is_same_v<CountT, std::int64_t>),
                "Can only currently use weights when Order <= 2.");

  static constexpr bool dbl_precision_weight = std::is_same_v<CountT, double>;

  /// Specifies a LUT for the moment_accums array.
  ///
  /// @note
  /// We put the enum declaration inside a struct so that the enumerator names
  /// must be namespace qualified. We DON'T use scoped enums because we want
  /// the enums to be interchangeable with integers
  struct LUT {
    enum {
      mean = 0 + dbl_precision_weight,
      cur_M2 = 1 + dbl_precision_weight,
      cur_M3 = 2 + dbl_precision_weight
    };
  };

public:  // interface
  // either StatDataView<1,Order> OR StatDataView<0,1+Order>
  using DataView =
      StatDataView<!dbl_precision_weight, Order + dbl_precision_weight>;

  static constexpr bool requires_weight = dbl_precision_weight;

  // this is just here so we can maintain consistency with histogram stat
  static constexpr bool alt_data_mapping = false;

  /// Return a description of the properties (int64 followed by doubles
  static PropDescr get_prop(int index) noexcept {
    if (index == 0) {
      return {"weight", std::is_same_v<CountT, double>, 1};
    } else if (index == 1) {
      return {"mean", true, 1};
    } else if ((Order >= 2) && (index == 2)) {
      return {"variance*weight", true, 1};
    } else if ((Order >= 3) && (index == 3)) {
      return {"cmoment3*count", true, 1};
    }
    return {};
  }

  CentralMomentStatistic(void* arg) { require(arg == nullptr, "invalid arg"); }

  static inline void add_entry(std::size_t spatial_idx, double val,
                               DataView& data) noexcept {
    if constexpr (std::is_same_v<CountT, std::int64_t>) {
      std::int64_t& count = data.get_i64(spatial_idx, 0);
      count++;
      double val_minus_last_mean = val - data.get_f64(spatial_idx, LUT::mean);
      double delta = val_minus_last_mean;
      double delta_div_n = delta / count;
      data.get_f64(spatial_idx, LUT::mean) += delta_div_n;
      if constexpr (Order > 1) {
        double val_minus_cur_mean = val - data.get_f64(spatial_idx, LUT::mean);
        double delta2_nm1_div_n = val_minus_last_mean * val_minus_cur_mean;
        if constexpr (Order > 2) {
          data.get_f64(spatial_idx, LUT::cur_M3) +=
              (delta2_nm1_div_n * delta_div_n * (count - 2) -
               3 * data.get_f64(spatial_idx, LUT::cur_M2) * delta_div_n);
        }
        data.get_f64(spatial_idx, LUT::cur_M2) += delta2_nm1_div_n;
      }
    } else {
      error("This version of the function won't work!");
    }
  }

  static inline void add_entry(std::size_t spatial_idx, double val,
                               double weight, DataView& data) noexcept {
    if constexpr (std::is_same_v<CountT, std::int64_t>) {
      /// we ignore the weight
      add_entry(spatial_idx, val, data);
    } else {
      double& weight_sum = data.get_f64(spatial_idx, 0);
      weight_sum += weight;
      double delta = val - data.get_f64(spatial_idx, LUT::mean);
      data.get_f64(spatial_idx, LUT::mean) +=
          (delta * weight) / (weight_sum + (weight_sum == 0));
      if constexpr (Order > 1) {
        double val_minus_cur_mean = val - data.get_f64(spatial_idx, LUT::mean);
        data.get_f64(spatial_idx, LUT::cur_M2) +=
            (weight * delta * val_minus_cur_mean);
      }
    }
  }

  /// Updates the values of `d_primary` to include contributions from `d_other`
  static void consolidate_helper_(std::size_t spatial_idx, DataView& d_primary,
                                  const DataView& d_other) noexcept {
    CountT& count = d_primary.template get<CountT>(spatial_idx, 0);
    CountT& o_count = d_other.template get<CountT>(spatial_idx, 0);

    if (count == 0) {
      d_primary.overwrite_register_from_other(d_other, spatial_idx);
    } else if (o_count == 0) {
      // do nothing
    } else if ((count == 1) && std::is_same_v<CountT, std::int64_t>) {
      // set temp equal to the value of the mean, currently held by `this`
      // (since the count is 1, this it exactly equal to the value of the sole
      // entry previously encountered by `this`)
      double temp = d_primary.get_f64(spatial_idx, LUT::mean);
      // overwrite the value of `this` with the contents of other
      d_primary.overwrite_register_from_other(d_other, spatial_idx);
      // add the value of the entry
      add_entry(spatial_idx, temp, d_primary);

    } else if ((o_count == 1) && std::is_same_v<CountT, std::int64_t>) {
      // equiv to adding a single entry to *this
      add_entry(spatial_idx, d_other.get_f64(spatial_idx, LUT::mean),
                d_primary);
    } else {  // general case
      double totcount = count + o_count;
      if constexpr (Order > 1) {
        double delta = (d_other.get_f64(spatial_idx, LUT::mean) -
                        d_primary.get_f64(spatial_idx, LUT::mean));
        double delta2_nprod_div_ntot =
            (delta * delta) * (count * o_count / totcount);
        if constexpr (Order > 2) {
          double term1 = delta2_nprod_div_ntot * (o_count - count);
          double term2 =
              3 * ((count * d_other.get_f64(spatial_idx, LUT::cur_M2)) -
                   (o_count * d_primary.get_f64(spatial_idx, LUT::cur_M2)));
          d_primary.get_f64(spatial_idx, LUT::cur_M3) =
              (d_primary.get_f64(spatial_idx, LUT::cur_M3) +
               d_other.get_f64(spatial_idx, LUT::cur_M3) +
               (delta * (term1 + term2)) / totcount);
        }
        d_primary.get_f64(spatial_idx, LUT::cur_M2) =
            (d_primary.get_f64(spatial_idx, LUT::cur_M2) +
             d_other.get_f64(spatial_idx, LUT::cur_M2) + delta2_nprod_div_ntot);
      }
      d_primary.get_f64(spatial_idx, LUT::mean) = consolidate_mean_(
          d_primary.get_f64(spatial_idx, LUT::mean), count,
          d_other.get_f64(spatial_idx, LUT::mean), o_count, totcount);
      count = totcount;
    }
  }

  /// Updates the values of `d_primary` to include contributions from `d_other`
  static void consolidate(DataView& d_primary,
                          const DataView& d_other) noexcept {
    const std::size_t n_spatial_bins = d_primary.num_registers();
    for (std::size_t i = 0; i < n_spatial_bins; i++) {
      consolidate_helper_(i, d_primary, d_other);
    }
  }
};

/// Implements a Statistic for measuring moments about the origin that can
/// that used to specialize Accumulator
///
/// @tparam Order Specifies the highest order moment that is computed. We also
///    compute all lower order moments
/// @tparam CountT Specifies the type of the counts. When this is int64_t
///    the class does standard stuff. When it is double, the class computes
///    the weighted moments
///
/// As a historical note, we originally had separate accumulators for mean,
/// variance, and weighted mean.
template <int Order, typename CountT = int64_t>
class OriginMomentStatistic {
  static_assert(1 <= Order, "at the moment we only allow 1 <= Order");
  static_assert(std::is_same_v<CountT, std::int64_t> ||
                    std::is_same_v<CountT, double>,
                "invalid type was used.");

  static constexpr bool dbl_precision_weight = std::is_same_v<CountT, double>;

public:  // interface
  // either StatDataView<1,Order> OR StatDataView<0,1+Order>
  using DataView =
      StatDataView<!dbl_precision_weight, Order + dbl_precision_weight>;

  static constexpr bool requires_weight = dbl_precision_weight;

  // this is just here so we can maintain consistency with histogram stat
  static constexpr bool alt_data_mapping = false;

  /// Return a description of the properties (int64 followed by doubles
  static PropDescr get_prop(int index) noexcept {
    if (index == 0) {
      return {"weight", std::is_same_v<CountT, double>, 1};
    } else if (index == 1) {
      return {"omoment1", true, 1};
    } else if ((Order >= 2) && (index == 2)) {
      return {"omoment2", true, 1};
    } else if ((Order >= 3) && (index == 3)) {
      return {"omoment3", true, 1};
    } else if ((Order >= 4) && (index == 4)) {
      return {"omoment4", true, 1};
    }
    return {};
  }

  OriginMomentStatistic(void* arg) { require(arg == nullptr, "invalid arg"); }

  static inline void add_entry(std::size_t spatial_idx, double val,
                               DataView& data) noexcept {
    if constexpr (std::is_same_v<CountT, std::int64_t>) {
      std::int64_t& count = data.get_i64(spatial_idx, 0);
      count++;
      double val_raised_to_ip1 = 1;
      for (int i = 0; i < Order; i++) {
        val_raised_to_ip1 *= val;
        double delta = val_raised_to_ip1 - data.get_f64(spatial_idx, i);
        double delta_div_n = delta / count;
        data.get_f64(spatial_idx, i) += delta_div_n;
      }
    } else {
      error("This version of the function won't work!");
    }
  }

  static inline void add_entry(std::size_t spatial_idx, double val,
                               double weight, DataView& data) noexcept {
    if constexpr (std::is_same_v<CountT, std::int64_t>) {
      /// we ignore the weight
      add_entry(val);
    } else {
      double& weight_sum = data.get_f64(spatial_idx, 0);
      weight_sum += weight;
      double val_raised_to_ip1 = 1;
      for (int i = 0; i < Order; i++) {
        val_raised_to_ip1 *= val;
        double delta = val_raised_to_ip1 - data.get_f64(spatial_idx, i + 1);
        data.get_f64(spatial_idx, i + 1) +=
            (delta * weight) / (weight_sum + (weight_sum == 0));
      }
    }
  }

  /// Updates the values of `d_primary` to include contributions from `d_other`
  static void consolidate_helper_(std::size_t spatial_idx, DataView& d_primary,
                                  const DataView& d_other) noexcept {
    CountT& count = d_primary.template get<CountT>(spatial_idx, 0);
    CountT& o_count = d_other.template get<CountT>(spatial_idx, 0);

    // when CountT is a double, we need to offset the indices for accessing all
    // of the moments by a value of 1
    const int offset = std::is_same_v<CountT, double>;

    if (count == 0) {
      d_primary.overwrite_register_from_other(d_other, spatial_idx);
    } else if (o_count == 0) {
      // do nothing
    } else if ((count == 1) && std::is_same_v<CountT, std::int64_t>) {
      // set temp equal to the value of the mean, currently held by `this`
      // (since the count is 1, this it exactly equal to the value of the sole
      // entry previously encountered by d_primary)
      double temp = d_primary.get_f64(spatial_idx, 0 + offset);
      // overwrite the value of `this` with the contents of other
      d_primary.overwrite_register_from_other(d_other, spatial_idx);
      // add the value of the entry
      add_entry(spatial_idx, temp, d_primary);

    } else if ((o_count == 1) && std::is_same_v<CountT, std::int64_t>) {
      // equiv to adding a single entry to d_primary
      double temp = d_other.get_f64(spatial_idx, 0 + offset);
      add_entry(spatial_idx, temp, d_primary);

    } else {  // general case
      double totcount = count + o_count;
      for (int i = 0; i < Order; i++) {
        d_primary.get_f64(spatial_idx, i + offset) = consolidate_mean_(
            d_primary.get_f64(spatial_idx, i + offset), count,
            d_other.get_f64(spatial_idx, i + offset), o_count, totcount);
      }
      count = totcount;
    }
  }

  /// Updates the values of `d_primary` to include contributions from `d_other`
  static void consolidate(DataView& d_primary,
                          const DataView& d_other) noexcept {
    const std::size_t n_spatial_bins = d_primary.num_registers();
    for (std::size_t i = 0; i < n_spatial_bins; i++) {
      consolidate_helper_(i, d_primary, d_other);
    }
  }
};

template <typename T>
class HistStatistic {
  static_assert(std::is_same_v<T, std::int64_t> || std::is_same_v<T, double>,
                "invalid type was used.");

  std::size_t n_data_bins_;
  std::vector<double> data_bin_edges_;

public:
  // either StatDataView<-1,0> OR StatDataView<0,-1>
  using DataView = StatDataView<-1 * std::is_same_v<T, std::int64_t>,
                                -1 * std::is_same_v<T, double>>;

  /// Returns the name of the stat computed by the accumulator
  static std::string stat_name() noexcept {
    return std::is_same_v<T, std::int64_t> ? "histogram" : "weightedhistogram";
  }

  /// Compile-time constant that specifies whether the add_entry overload with
  /// the weight argument must be used.
  static constexpr bool dbl_precision_weight = std::is_same_v<T, double>;
  static constexpr bool requires_weight = dbl_precision_weight;

  // this is just here so we can maintain historical consistency
  static constexpr bool alt_data_mapping = true;

  HistStatistic() noexcept : n_data_bins_(), data_bin_edges_() {}

  HistStatistic(void* arg) : n_data_bins_(), data_bin_edges_() {
    if (arg == nullptr) error("arg must not be a nullptr");

    BinSpecification* data_bins = static_cast<BinSpecification*>(arg);

    // initialize n_data_bins_
    require(data_bins->n_bins > 0, "There must be a positive number of bins.");
    n_data_bins_ = data_bins->n_bins;

    // initialize data_bin_edges_ (copy data from data_bin_edges)
    std::size_t len_data_bin_edges = n_data_bins_ + 1;
    data_bin_edges_.resize(len_data_bin_edges);
    // we should really confirm the data_bin_edges_ is monotonic
    for (std::size_t i = 0; i < len_data_bin_edges; i++) {
      data_bin_edges_[i] = data_bins->bin_edges[i];
    }
  }

  /// Add an entry (without a weight)
  inline void add_entry(std::size_t spatial_bin_index, double val,
                        DataView& data) noexcept {
    if constexpr (std::is_same_v<T, std::int64_t>) {
      std::size_t data_bin_index =
          identify_bin_index(val, data_bin_edges_.data(), n_data_bins_);
      if (data_bin_index < n_data_bins_) {
        std::size_t i = data_bin_index + spatial_bin_index * n_data_bins_;
        data.get_i64(spatial_bin_index, data_bin_index)++;
      }
    } else {
      error("a weight must be provided!");
    }
  }

  /// Add an entry (with a weight)
  inline void add_entry(std::size_t spatial_bin_index, double val,
                        double weight, DataView& data) noexcept {
    if constexpr (std::is_same_v<T, std::int64_t>) {
      // ignore the weight
      add_entry(spatial_bin_index, val);
    } else {
      std::size_t data_bin_index =
          identify_bin_index(val, data_bin_edges_.data(), n_data_bins_);
      if (data_bin_index < n_data_bins_) {
        data.get_f64(spatial_bin_index, data_bin_index) += weight;
      }
    }
  }

  /// Updates the values of `d_primary` to include contributions from `d_other`
  static void consolidate(DataView& d_primary,
                          const DataView& d_other) noexcept {
    d_primary.inplace_add(d_other);
  }

  /// Return a description of the properties (int64 followed by doubles
  PropDescr get_prop(int index) const noexcept {
    if (index == 0) return {"weight", dbl_precision_weight, int(n_data_bins_)};
    return {};
  }
};

/// helper function that applies the given function object to each of a
/// statitic's property descriptions for all properties that have a particular
/// type ``T``
template <typename T, typename Stat, typename UnaryFunc>
void for_each_typed_prop_(const Stat& statistic, UnaryFunc fn) {
  static_assert(std::is_same_v<T, std::int64_t> || std::is_same_v<T, double>,
                "invalid type was provided.");
  int prop_index = 0;
  while (true) {
    PropDescr descr = statistic.get_prop(prop_index);
    if (descr.name == nullptr) break;
    if (std::is_same_v<T, double> == descr.is_f64) fn(descr);
    prop_index++;
  }
}

/// An Accumulator wraps a Statistic class
///
/// A statistic class implements the logic to compute the logic. The
/// corresponding accumulator also tracks the intermediate data computed by
/// the statistic.
template <typename Stat>
struct Accumulator {
  using DataView = typename Stat::DataView;

  /// Compile-time constant that specifies whether the add_entry overload with
  /// the weight argument must be used.
  static constexpr bool requires_weight = Stat::requires_weight;

private:  // attributes
  std::size_t n_spatial_bins_;
  Stat statistic_;
  DataView data_;

private:  // helper functions
  template <typename T>
  std::size_t num_type_vals_per_spatial_bin_() const {
    std::size_t count = 0;
    for_each_typed_prop_<T>(statistic_, [&](PropDescr d) { count += d.count; });
    return count;
  }

public:  // interface
  /// Returns the name of the stat computed by the accumulator
  static std::string stat_name() noexcept { return Stat::stat_name(); }

  Accumulator() = default;

  Accumulator(std::size_t n_spatial_bins, void* other_arg) noexcept
      : n_spatial_bins_(n_spatial_bins), statistic_(other_arg) {
    require(n_spatial_bins > 0, "n_spatial_bins must be positive");
    // in the future, we could defer initialization of data_
    std::size_t n_i64 = this->num_type_vals_per_spatial_bin_<std::int64_t>();
    std::size_t n_f64 = this->num_type_vals_per_spatial_bin_<double>();
    std::exchange(data_, DataView(n_spatial_bins_, n_i64, n_f64));
  }

  /// copy constructor.
  ///
  /// @note
  /// we may be able to drop this in the future
  Accumulator(const Accumulator& o)
      : n_spatial_bins_(o.n_spatial_bins_),
        statistic_(o.statistic_),
        data_(o.data_.clone()) {}

  /// copy assignment
  ///
  /// @note
  /// we may be able to drop this in the future
  Accumulator& operator=(const Accumulator& other) {
    if (this != &other) {
      this->n_spatial_bins_ = other.n_spatial_bins_;
      this->statistic_ = other.statistic_;
      this->data_ = other.data_.clone();
    }
    return *this;
  }

  Accumulator(Accumulator&&) = default;
  Accumulator& operator=(Accumulator&&) = default;

  /// reset the contents (it looks as though we just initialized)
  ///
  /// if we ever introduce a statistic where the default values aren't zero,
  /// then we will need to revisit this!
  void purge() noexcept { data_.zero_fill(); }

  inline void add_entry(std::size_t spatial_bin_index, double val) noexcept {
    statistic_.add_entry(spatial_bin_index, val, this->data_);
  }

  inline void add_entry(std::size_t spatial_bin_index, double val,
                        double weight) noexcept {
    statistic_.add_entry(spatial_bin_index, val, weight, this->data_);
  }

  /// Updates the values of `*this` to include the values from `other`
  inline void consolidate_with_other(const Accumulator& other) noexcept {
    require(other.n_spatial_bins_ == n_spatial_bins_,
            "a mismatch was encountered during consolidation");
    Stat::consolidate(this->data_, other.data_);
  }

  /// Return the Floating Point Value Properties
  ///
  /// This is a vector holding pairs of the value name and the number of value
  /// of value entries in a given spatial bin.
  ///
  /// @todo
  /// Consider consolidating this function with i64_val_props. Also consider
  /// returning a std::vector<PropDescr> instead of a vector of pairs
  std::vector<std::pair<std::string, std::size_t>> flt_val_props()
      const noexcept {
    std::vector<std::pair<std::string, std::size_t>> out;
    for_each_typed_prop_<double>(statistic_, [&](PropDescr d) {
      out.push_back(std::make_pair(d.name, d.count));
    });
    return out;
  }

  /// Return the Int64 Value Properties
  ///
  /// This is a vector holding pairs of the integer value name and the number
  /// of value entries in a given spatial bin.
  ///
  /// @todo
  /// Consider consolidating this function with f64_val_props. Also consider
  /// returning a std::vector<PropDescr> instead of a vector of pairs
  std::vector<std::pair<std::string, std::size_t>> i64_val_props()
      const noexcept {
    std::vector<std::pair<std::string, std::size_t>> out;
    for_each_typed_prop_<std::int64_t>(statistic_, [&](PropDescr d) {
      out.push_back(std::make_pair(d.name, d.count));
    });
    return out;
  }

  /// Copies values of each scalar accumulator to a pre-allocated buffer
  template <typename T>
  void copy_vals(T* out_vals) const noexcept {
    const std::size_t num_dtype_vals = num_type_vals_per_spatial_bin_<T>();

    for (std::size_t i = 0; i < n_spatial_bins_; i++) {
      for (std::size_t j = 0; j < num_dtype_vals; j++) {
        if constexpr (Stat::alt_data_mapping) {
          out_vals[j + i * num_dtype_vals] = data_.template get<T>(i, j);
        } else {
          out_vals[i + j * n_spatial_bins_] = data_.template get<T>(i, j);
        }
      }
    }
  }

  /// Overwrites the values within each scalar accumulator using data from an
  /// external buffer
  ///
  /// This is primarily meant to be passed an external buffer whose values were
  /// initialized by the copy_vals method.
  template <typename T>
  void import_vals(const T* in_vals) noexcept {
    const std::size_t num_dtype_vals = num_type_vals_per_spatial_bin_<T>();

    for (std::size_t i = 0; i < n_spatial_bins_; i++) {
      for (std::size_t j = 0; j < num_dtype_vals; j++) {
        if constexpr (Stat::alt_data_mapping) {
          data_.template get<T>(i, j) = in_vals[j + i * num_dtype_vals];
        } else {
          data_.template get<T>(i, j) = in_vals[i + j * n_spatial_bins_];
        }
      }
    }
  }

  std::size_t n_spatial_bins() const noexcept { return n_spatial_bins_; }
};

using HistogramAccumCollection = Accumulator<HistStatistic<std::int64_t>>;
using WeightedHistogramAccumCollection = Accumulator<HistStatistic<double>>;

#endif /* ACCUMULATORS_H */
