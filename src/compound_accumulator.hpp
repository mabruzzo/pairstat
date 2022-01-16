#ifndef COMPOUND_ACCUMULATOR_H
#define COMPOUND_ACCUMULATOR_H

#include <string>
#include <tuple>
#include <type_traits>
#include <utility> // std::pair
#include <vector>

namespace detail{

  template<typename Tup, class Func, std::size_t countdown>
  struct for_each_tuple_entry_{
    static inline void evaluate(Tup& tuple, Func& f) noexcept{
      auto& elem = std::get<std::tuple_size_v<Tup> - countdown>(tuple);
      f(elem);
      for_each_tuple_entry_<Tup, Func, countdown-1>::evaluate(tuple, f);
    }
  };

  template<typename Tup, class Func>
  struct for_each_tuple_entry_<Tup,Func,0>{
    static inline void evaluate(Tup& tuple, Func& f) noexcept{ }
  };
} /* namespace detail */

/// Apply a functor to all elements of a tuple
template<class T, class UnaryFunction>
constexpr inline void for_each_tuple_entry(T& tuple, UnaryFunction f){
  detail::for_each_tuple_entry_<T, UnaryFunction,
                                std::tuple_size_v<T>>::evaluate(tuple, f);
}

namespace detail {

  /// typesafe function that copies data from an AccumCollection to a pointer
  ///
  /// We enable the functions based on the return type
  template<typename AccumCollec, typename T>
  typename std::enable_if<std::is_same<T, int64_t>::value, void>::type
    copy_data_(const AccumCollec& accum_collec, T* dest) noexcept
  { accum_collec.copy_i64_vals(dest); }

  template<typename AccumCollec, typename T>
  typename std::enable_if<std::is_same<T, double>::value, void>::type
    copy_data_(const AccumCollec& accum_collec, T* dest) noexcept
  { accum_collec.copy_flt_vals(dest); }


} /* namespace detail */

template<typename T>
struct CopyValsHelper_{
  CopyValsHelper_(T* data_ptr)
    : data_ptr_(data_ptr), offset_(0)
  { }

  template<class AccumCollec>
  void operator()(const AccumCollec& accum_collec) noexcept{

    detail::copy_data_(accum_collec, data_ptr_ + offset_);

    std::vector<std::pair<std::string,std::size_t>> val_props;
    if (std::is_same<T, int64_t>::value){
      val_props = accum_collec.i64_val_props();
    } else {
      val_props = accum_collec.flt_val_props();
    }

    std::size_t n_spatial_bins = accum_collec.n_spatial_bins();
    for (const auto& [quan_name,elem_per_spatial_bin] : val_props) {
      offset_ += n_spatial_bins * elem_per_spatial_bin;
    }
  }

  T* data_ptr_;
  std::size_t offset_;
};


template<typename AccumCollectionTuple>
class CompoundAccumCollection{

  /// @class    CompoundAccumCollection
  ///
  /// @brief Supports multiple accumulators at the same time. This is something
  ///    of a stopgap solution.

public:

  static constexpr std::size_t n_accum =
    std::tuple_size_v<AccumCollectionTuple>;

  static_assert(n_accum > 1,
                "CompoundAccumCollection must be composed of 2+ accumulators.");

  CompoundAccumCollection() = delete;

  CompoundAccumCollection(const CompoundAccumCollection&) = default;

  CompoundAccumCollection(AccumCollectionTuple &&accum_collec_tuple) noexcept
    : accum_collec_tuple_(accum_collec_tuple)
  {}

  inline void add_entry(std::size_t spatial_bin_index, double val) noexcept{
    for_each_tuple_entry(accum_collec_tuple_,
                         [=](auto& e){ e.add_entry(spatial_bin_index, val); });
  }

  /// Updates the values of `*this` to include the values from `other`
  inline void consolidate_with_other(const CompoundAccumCollection& other)
    noexcept
  { error("Not Implemented Yet"); }

  /// Copies the int64_t values of each accumulator to an external buffer
  void copy_i64_vals(int64_t *out_vals) noexcept {
    for_each_tuple_entry(accum_collec_tuple_, CopyValsHelper_(out_vals));
  }

  /// Copies the floating point values of each accumulator to an external buffer
  void copy_flt_vals(double *out_vals) noexcept {
    for_each_tuple_entry(accum_collec_tuple_, CopyValsHelper_(out_vals));
  }

  /// Dummy method that needs to be defined to match interface
  static std::vector<std::pair<std::string,std::size_t>> flt_val_props()
    noexcept
  { error("Not Implemented"); }

  /// Dummy method that needs to be defined to match interface
  std::vector<std::pair<std::string,std::size_t>> i64_val_props() noexcept
  { error("Not Implemented"); }

  /// Dummy method that needs to be defined to match interface
  void import_flt_vals(const double *in_vals) noexcept
  { error("Not Implemented"); }

  /// Dummy method that needs to be defined to match interface
  void import_i64_vals(const int64_t *in_vals) noexcept
  { error("Not Implemented"); }

private:
  AccumCollectionTuple accum_collec_tuple_;
};

#endif /* COMPOUND_ACCUMULATOR_H */
