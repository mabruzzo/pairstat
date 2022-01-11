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

public:

  static constexpr std::size_t n_accum =
    std::tuple_size_v<AccumCollectionTuple>;

  static_assert(n_accum > 1,
                "CompoundAccumCollection must be composed of 2+ accumulators.");

  CompoundAccumCollection() = delete;

  CompoundAccumCollection(AccumCollectionTuple &&accum_collec_tuple) noexcept
    : accum_collec_tuple_(accum_collec_tuple)
  {}

  inline void add_entry(std::size_t spatial_bin_index, double val) noexcept{
    for_each_tuple_entry(accum_collec_tuple_,
                         [=](auto& e){ e.add_entry(spatial_bin_index, val); });
  }

  void copy_i64_vals(int64_t *out_vals) noexcept {
    for_each_tuple_entry(accum_collec_tuple_, CopyValsHelper_(out_vals));
  }

  void copy_flt_vals(double *out_vals) noexcept {
    for_each_tuple_entry(accum_collec_tuple_, CopyValsHelper_(out_vals));
  }

private:
  AccumCollectionTuple accum_collec_tuple_;
};

#endif /* COMPOUND_ACCUMULATOR_H */
