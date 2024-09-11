#ifndef ACCUMULATORS_H
#define ACCUMULATORS_H

#include <algorithm>  // std::fill
#include <cstdint>    // std::int64_t
#include <string>
#include <type_traits>  // std::is_same_v
#include <utility>      // std::pair
#include <vector>

#include "vsf.hpp" // BinSpecification
#include "utils.hpp"  // error

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

// The accumulator structs must all satisfy the following properties:
// - static method called ``flt_val_names`` that returns the names of each
//   floating point value
// - instance method called ``get_flt_val`` that returns the stored floating
//   point value corresponding to the name returned by flt_val_names
// - must have a default constructor
// - must define the ``add_entry`` instance method that updates the
//   statistic(s) that are being accumulated.
// - must currently define the count attribute (which tracks the number of
//   entries that have been added to the accumulator so far).


/// Implements an accumulator (for measuring central moments) that can be
/// used to simplify ScalarAccumCollection
///
/// As a historical note, we originally had separate accumulators for mean
/// and variance.
///
/// @note
/// The 1st and 2nd second moments exactly the same as mean and variance.
/// While the 3rd and 4th moments are related to skew and kurtosis, they are
/// not exactly the same.
///
/// @note
/// In the future, to generalize to higher order moments, we can look at
/// https://zenodo.org/records/1232635
template <int Order>
class CentralMomentAccum {
  static_assert((1 <= Order) && (Order <= 2),
                "at the moment we only allow 1 <= Order <=2");

  /// Specifies a LUT for the moment_accums array.
  ///
  /// @note
  /// We put the enum declaration inside a struct so that the enumerator names
  /// must be namespace qualified. We DON'T use scoped enums because we want
  /// the enums to be interchangeable with integers
  struct LUT {
    enum { mean = 0, cur_M2 = 1 };
  };

public:  // interface
  /// Returns the name of the stat computed by the accumulator
  static std::string stat_name() noexcept {
    if constexpr (Order == 1) {
      return "mean";
    } else {
      return "variance";
    }
  }

  static std::vector<std::string> i64_val_names() noexcept { return {"count"}; }

  static std::vector<std::string> flt_val_names() noexcept {
    if constexpr (Order == 1) {
      return {"mean"};
    } else {
      return {"mean", "variance*count"};
    }
  }

  static constexpr bool requires_weight = false;

  template <typename T>
  const T& access(std::size_t i) const noexcept {
    if constexpr (std::is_same_v<T, std::int64_t>) {
      if (i != 0) error("only has 1 integer value");
      return this->count;
    } else if constexpr (std::is_same_v<T, double>) {
      if (i >= Order) error("trying to access a non-existent float_val");
      return moment_accums[i];
    } else {
      static_assert(dummy_false_v_<T>,
                    "template T must be double or std::int64_t");
    }
  }

  template <typename T>
  T& access(std::size_t i) noexcept {
    const T& out =
        const_cast<const CentralMomentAccum<Order>*>(this)->access<T>(i);
    return const_cast<T&>(out);
  }

  CentralMomentAccum() : count(0) {
    std::fill(moment_accums, moment_accums + Order, 0);
  }

  inline void add_entry(double val) noexcept {
    count++;
    double val_minus_last_mean = val - moment_accums[LUT::mean];
    moment_accums[LUT::mean] += (val_minus_last_mean) / count;
    if constexpr (Order > 1) {
      double val_minus_cur_mean = val - moment_accums[LUT::mean];
      moment_accums[LUT::cur_M2] += val_minus_last_mean * val_minus_cur_mean;
    }
  }

  /// we ignore the weight
  inline void add_entry(double val, double weight) noexcept { add_entry(val); }

  inline void consolidate_with_other(
      const CentralMomentAccum<Order>& other) noexcept {
    if (this->count == 0) {
      (*this) = other;
    } else if (other.count == 0) {
      // do nothing
    } else if (this->count == 1) {
      // set temp equal to the value of the mean, currently held by `this`
      // (since the count is 1, this it exactly equal to the value of the sole
      // entry previously encountered by `this`)
      double temp = this->moment_accums[LUT::mean];
      // overwrite the value of `this` with the contents of other
      (*this) = other;
      // add the value of the entry
      this->add_entry(temp);

    } else if (other.count == 1) {  // equiv to adding a single entry to *this
      this->add_entry(other.moment_accums[LUT::mean]);
    } else {  // general case
      double totcount = this->count + other.count;
      if constexpr (Order > 1) {
        double delta =
            (other.moment_accums[LUT::mean] - this->moment_accums[LUT::mean]);
        double delta2 = delta * delta;
        this->moment_accums[LUT::cur_M2] =
            (this->moment_accums[LUT::cur_M2] +
             other.moment_accums[LUT::cur_M2] +
             delta2 * (this->count * other.count / totcount));
      }
      this->moment_accums[LUT::mean] = consolidate_mean_(
          this->moment_accums[LUT::mean], this->count,
          other.moment_accums[LUT::mean], other.count, totcount);
      this->count = totcount;
    }
  }

public:  // attributes
  // number of entries included (so far)
  int64_t count;

  /// holds the accumulator variables for each moment. When present the indices
  /// map to the follwing quantities:
  ///  * 0: mean, the current mean
  ///  * 1: cur_M2, sum of differences from the current mean
  double moment_accums[Order];
};

struct WeightedMeanAccum {
public:  // interface
  /// Returns the name of the stat computed by the accumulator
  static std::string stat_name() noexcept { return "weightedmean"; }

  static std::vector<std::string> i64_val_names() noexcept { return {}; }

  static std::vector<std::string> flt_val_names() noexcept {
    return {"weight_sum", "mean"};
  }

  static constexpr bool requires_weight = true;

  template <typename T>
  const T& access(std::size_t i) const noexcept {
    if constexpr (std::is_same_v<T, std::int64_t>) {
      error("WeightedMeanAccum only has 1 integer value");
    } else if constexpr (std::is_same_v<T, double>) {
      switch (i) {
        case 0:
          return weight_sum;
        case 1:
          return mean;
        default:
          error("WeightedMeanAccum only has 2 float_vals");
      }
    } else {
      static_assert(dummy_false_v_<T>,
                    "template T must be double or std::int64_t");
    }
  }

  template <typename T>
  T& access(std::size_t i) noexcept {
    return const_cast<T&>(
        const_cast<const WeightedMeanAccum*>(this)->access<T>(i));
  }

  WeightedMeanAccum() : weight_sum(0.0), mean(0.0) {}

  inline void add_entry(double val) noexcept {
    error("This version of the function won't work!");
  }

  inline void add_entry(double val, double weight) noexcept {
    weight_sum += weight;
    double val_minus_last_mean = val - mean;
    // we could use the following line if we wanted to allow weight=0
    // mean += (val_minus_last_mean * weight) / (weight_sum + (weight_sum == 0);
    mean += (val_minus_last_mean * weight) / weight_sum;
  }

  inline void consolidate_with_other(const WeightedMeanAccum& other) noexcept {
    if (this->weight_sum == 0.0) {
      (*this) = other;
    } else if (other.weight_sum == 0.0) {
      // do nothing
    } else {  // general case
      double tot_weight = this->weight_sum + other.weight_sum;
      this->mean = consolidate_mean_(this->mean, this->weight_sum, other.mean,
                                     other.weight_sum, tot_weight);
      this->weight_sum = tot_weight;
    }
  }

public:  // attributes
  // the total weight (so far)
  double weight_sum;
  // current mean
  double mean;
};

template <typename Accum, typename T>
inline std::size_t num_type_vals_per_bin_() {
  if constexpr (std::is_same_v<T, std::int64_t>) {
    return Accum::i64_val_names().size();
  } else if constexpr (std::is_same_v<T, double>) {
    return Accum::flt_val_names().size();
  } else {
    static_assert(dummy_false_v_<T>,
                  "template T must be double or std::int64_t");
  }
}

template <typename Accum>
struct ScalarAccumCollection {
  /// Returns the name of the stat computed by the accumulator
  static std::string stat_name() noexcept { return Accum::stat_name(); }

  /// Compile-time constant that specifies whether the add_entry overload with
  /// the weight argument must be used.
  static constexpr bool requires_weight = Accum::requires_weight;

  ScalarAccumCollection() noexcept : accum_list_() {}

  ScalarAccumCollection(std::size_t n_spatial_bins, void* other_arg) noexcept
      : accum_list_(n_spatial_bins) {
    if (n_spatial_bins == 0) {
      error("n_spatial_bins must be positive");
    }
    if (other_arg != nullptr) {
      error("other_arg must be nullptr");
    }
  }

  /// reset the contents (it looks as though we just initialized)
  void purge() noexcept {
    std::fill(accum_list_.begin(), accum_list_.end(), Accum());
  }

  inline void add_entry(std::size_t spatial_bin_index, double val) noexcept {
    accum_list_[spatial_bin_index].add_entry(val);
  }

  inline void add_entry(std::size_t spatial_bin_index, double val,
                        double weight) noexcept {
    accum_list_[spatial_bin_index].add_entry(val, weight);
  }

  /// Updates the values of `*this` to include the values from `other`
  inline void consolidate_with_other(
      const ScalarAccumCollection& other) noexcept {
    std::size_t num_bins = accum_list_.size();
    if (other.accum_list_.size() != num_bins) {
      error("There seemed to be a mismatch during consolidation");
    }
    for (std::size_t i = 0; i < num_bins; i++) {
      accum_list_[i].consolidate_with_other(other.accum_list_[i]);
    }
  }

  /// Return the Floating Point Value Properties
  ///
  /// This is a vector holding pairs of the value name and the number of value
  /// entries per spatial bin.
  ///
  /// For Scalar Accumulators, each value only stores 1 entry per spatial bin
  static std::vector<std::pair<std::string, std::size_t>>
  flt_val_props() noexcept {
    std::vector<std::string> flt_val_names = Accum::flt_val_names();

    std::vector<std::pair<std::string, std::size_t>> out;
    for (std::size_t i = 0; i < flt_val_names.size(); i++) {
      out.push_back(std::make_pair(flt_val_names[i], 1));
    }
    return out;
  }

  /// Return the Int64 Value Properties
  ///
  /// This is a vector holding pairs of the integer value name and the number
  /// of value entries per spatial bin.
  ///
  /// This is currently the same for all scalar accumulators
  static std::vector<std::pair<std::string, std::size_t>>
  i64_val_props() noexcept {
    std::vector<std::string> i64_val_names = Accum::i64_val_names();

    std::vector<std::pair<std::string, std::size_t>> out;
    for (std::size_t i = 0; i < i64_val_names.size(); i++) {
      out.push_back(std::make_pair(i64_val_names[i], 1));
    }
    return out;
  }

  /// Copies values of each scalar accumulator to a pre-allocated buffer
  template <typename T>
  void copy_vals(T* out_vals) const noexcept {
    const std::size_t num_dtype_vals = num_type_vals_per_bin_<Accum, T>();
    const std::size_t n_bins = accum_list_.size();

    for (std::size_t i = 0; i < n_bins; i++) {
      for (std::size_t j = 0; j < num_dtype_vals; j++) {
        out_vals[i + j * n_bins] = accum_list_[i].template access<T>(j);
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
    const std::size_t num_flt_vals = num_type_vals_per_bin_<Accum, T>();
    const std::size_t n_bins = accum_list_.size();

    for (std::size_t i = 0; i < n_bins; i++) {
      for (std::size_t j = 0; j < num_flt_vals; j++) {
        accum_list_[i].template access<T>(j) = in_vals[i + j * n_bins];
      }
    }
  }

  std::size_t n_spatial_bins() const noexcept { return accum_list_.size(); }

private:
  std::vector<Accum> accum_list_;
};

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

/// Implements a histogram accumulator
template <typename T>
class GenericHistogramAccumCollection {
  static_assert(std::is_same_v<T, std::int64_t> || std::is_same_v<T, double>,
                "invalid type was used.");

public:
  /// Returns the name of the stat computed by the accumulator
  static std::string stat_name() noexcept {
    return std::is_same_v<T, std::int64_t> ? "histogram" : "weightedhistogram";
  }

  /// Compile-time constant that specifies whether the add_entry overload with
  /// the weight argument must be used.
  static constexpr bool requires_weight = std::is_same_v<T, double>;

  GenericHistogramAccumCollection() noexcept
      : n_spatial_bins_(), n_data_bins_(), bin_counts_(), data_bin_edges_() {}

  GenericHistogramAccumCollection(std::size_t n_spatial_bins,
                                  void* other_arg) noexcept
      : n_spatial_bins_(n_spatial_bins),
        n_data_bins_(),
        bin_counts_(),
        data_bin_edges_() {
    if (n_spatial_bins == 0) {
      error("n_spatial_bins must be positive");
    }
    if (other_arg == nullptr) {
      error("other_arg must not be a nullptr");
    }

    BinSpecification* data_bins = static_cast<BinSpecification*>(other_arg);

    // initialize n_data_bins_
    if (data_bins->n_bins == 0) {
      error("There must be a positive number of bins.");
    }
    n_data_bins_ = data_bins->n_bins;

    // initialize data_bin_edges_ (copy data from data_bin_edges)
    std::size_t len_data_bin_edges = n_data_bins_ + 1;
    data_bin_edges_.resize(len_data_bin_edges);
    // we should really confirm the data_bin_edges_ is monotonic
    for (std::size_t i = 0; i < len_data_bin_edges; i++) {
      data_bin_edges_[i] = data_bins->bin_edges[i];
    }

    // initialize the counts array
    bin_counts_.resize(n_data_bins_ * n_spatial_bins_, 0);
  }

  /// reset the contents (it looks as though we just initialized)
  void purge() noexcept {
    std::fill(bin_counts_.begin(), bin_counts_.end(), T(0));
  }

  /// Add an entry (without a weight)
  inline void add_entry(std::size_t spatial_bin_index, double val) noexcept {
    if constexpr (std::is_same_v<T, std::int64_t>) {
      std::size_t data_bin_index =
          identify_bin_index(val, data_bin_edges_.data(), n_data_bins_);
      if (data_bin_index < n_data_bins_) {
        std::size_t i = data_bin_index + spatial_bin_index * n_data_bins_;
        bin_counts_[i]++;
      }
    } else {
      error("a weight must be provided!");
    }
  }

  /// Add an entry (with a weight)
  inline void add_entry(std::size_t spatial_bin_index, double val,
                        double weight) noexcept {
    if constexpr (std::is_same_v<T, std::int64_t>) {
      // ignore the weight
      add_entry(spatial_bin_index, val);
    } else {
      std::size_t data_bin_index =
          identify_bin_index(val, data_bin_edges_.data(), n_data_bins_);
      if (data_bin_index < n_data_bins_) {
        std::size_t i = data_bin_index + spatial_bin_index * n_data_bins_;
        bin_counts_[i] += weight;
      }
    }
  }

  /// Updates the values of `*this` to include the values from `other`
  inline void consolidate_with_other(
      const GenericHistogramAccumCollection<T>& other) noexcept {
    if ((other.n_spatial_bins_ != n_spatial_bins_) ||
        (other.n_data_bins_ != n_data_bins_)) {
      error("There seemed to be a mismatch during consolidation");
    }
    // going to simply assume that contents of data_bin_edges_ are consistent

    const std::size_t stop = bin_counts_.size();

    for (std::size_t i = 0; i < stop; i++) {
      bin_counts_[i] += other.bin_counts_[i];
    }
  }

  /// Return the Floating Point Value Properties (if any)
  std::vector<std::pair<std::string, std::size_t>> flt_val_props()
      const noexcept {
    if (std::is_same_v<T, std::int64_t>) {
      return {};
    } else {
      return {{"bin_weights_", n_data_bins_}};
    }
  }

  /// Return the Int64 Value Properties
  ///
  /// This is a vector holding pairs of the integer value name and the number
  /// of value entries per spatial bin.
  std::vector<std::pair<std::string, std::size_t>> i64_val_props()
      const noexcept {
    if (std::is_same_v<T, std::int64_t>) {
      return {{"bin_counts_", n_data_bins_}};
    } else {
      return {};
    }
  }

  /// Copies the values of the accumulator to a pre-allocatd buffer
  template <typename Tbuf>
  void copy_vals(Tbuf* out_vals) const noexcept {
    if constexpr (std::is_same_v<Tbuf, T>) {
      for (std::size_t i = 0; i < bin_counts_.size(); i++) {
        out_vals[i] = bin_counts_[i];
      }
    } else if constexpr (std::is_same_v<Tbuf, std::int64_t> ||
                         std::is_same_v<Tbuf, double>) {
      // DO NOTHING!
    } else {
      static_assert(dummy_false_v_<Tbuf>,
                    "template Tbuf must be double or std::int64_t");
    }
  }

  /// Overwrites the accumulator's values using data from an external buffer
  ///
  /// This is primarily meant to be passed an external buffer whose values were
  /// initialized by the copy_vals method.
  template <typename Tbuf>
  void import_vals(const Tbuf* in_vals) noexcept {
    if constexpr (std::is_same_v<Tbuf, T>) {
      for (std::size_t i = 0; i < bin_counts_.size(); i++) {
        bin_counts_[i] = in_vals[i];
      }
    } else if constexpr (std::is_same_v<Tbuf, std::int64_t> ||
                         std::is_same_v<Tbuf, double>) {
      // DO NOTHING!
    } else {
      static_assert(dummy_false_v_<T>,
                    "template Tbuf must be double or std::int64_t");
    }
  }

  std::size_t n_spatial_bins() const noexcept { return n_spatial_bins_; }

private:
  std::size_t n_spatial_bins_;
  std::size_t n_data_bins_;

  // counts_ holds the histogram counts. It holds n_data_bins_*n_spatial_bins_
  // entries. The counts for the ith data bin in the jth spatial bin are stored
  // at index (i + j * n_data_bins_)
  std::vector<T> bin_counts_;

  std::vector<double> data_bin_edges_;
};

// for backwards compatability
using HistogramAccumCollection = GenericHistogramAccumCollection<std::int64_t>;
using WeightedHistogramAccumCollection =
    GenericHistogramAccumCollection<double>;

#endif /* ACCUMULATORS_H */
