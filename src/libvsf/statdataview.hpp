#ifndef STATDATAVIEW_HPP
#define STATDATAVIEW_HPP

#include <cstdint>      // std::int64_t
#include <cstring>      // std::memset
#include <type_traits>  // std::is_trivially_copyable_v
#include <utility>      // std::exchange

#include "utils.hpp"

namespace detail {

template <int N_i64, int N_f64>
struct ImplProps_ {
  static_assert((N_i64 == 0) || (N_i64 == 1) || (N_i64 == -1),
                "N_i64 can only be -1, 1, or 0");
  static_assert((N_f64 >= 0), "N_f64 must be -1 or a non-negative value");
  static_assert(!((N_i64 == -1) && (N_f64 != 0)),
                "when N_i64 is -1, N_f64 must be 0");
  static_assert(!((N_i64 != 0) && (N_f64 == -1)),
                "when N_f64 is -1, N_f64 must be 0");

  static_assert(!((N_i64 == 0) && (N_f64 == 0)),
                "we could add support for this case in the future!");
};

template <>
struct ImplProps_<-1, 0> {
  using element_type = std::int64_t;
  static constexpr bool variable_len = true;
};

template <>
struct ImplProps_<0, -1> {
  using element_type = double;
  static constexpr bool variable_len = true;
};

template <int N_f64>
struct ImplProps_<0, N_f64> {
  struct DummyType {
    double f64_vals[N_f64];
  };
  using element_type = DummyType;
  static constexpr bool variable_len = false;
};

template <int N_f64>
struct ImplProps_<1, N_f64> {
  struct DummyType {
    std::int64_t i64_vals[1];
    double f64_vals[N_f64];
  };
  using element_type = DummyType;
  static constexpr bool variable_len = false;
};
}  // namespace detail

/// This is a pointer-like class that is used to manage memory used inside of
/// an accumulator
///
/// At the moment, this has semantics like std::unique_ptr. We might want to
/// make the semantics a little more like std::shared_ptr in the future (but we
/// will cross that bridge if it becomes necessary)
///
/// @note
/// This is a view in a similar sense to a Kokkos View (although this isn't
/// very feature rich).
///
/// @par Challenges with Interleaved Datatypes
/// One of the goals of this class is to interleave the integer and floating
/// point values. In other words, we wanted memory to be organized as an
/// an array of structs (not a struct of arrays). To accomplish this, we made
/// use of Templates.
///
/// @par
/// We could achieve a similar effect with dynamic allocations (i.e. without
/// using templates). Doing this properly (i.e. respecting C++'s object model)
/// is a little tricky. I think we could do the following:
///   - internally we would allocate and store a pointer to storage for all of
///     the desired values. Immediately after allocation, we should call the
///     "placement ``new``" operation to initialize every single quantity.
///   - any time we try to access a value, we would compute the ptr to the
///     location in storage where the value is stored, and then return
///     ``std::launder(reinterpret_cast<T*>(ptr))``
///   - during destruction, it's not clear what we need to do. Since we are
///     using std::int64_t and double, we might not need to explicitly do
///     anything (they are implicitly created types). This is a little unclear.
///   - alternatively, we could make repeated use of std::start_lifetime_as (I
///     think that destruction requirements would make more sense)
///
/// @note
/// In the future, we may want to support wrapping memory managed by a
/// structured numpy array. This will definitely takes some care. There are
/// definitely a lot of challenges related to undefined behavior and C++'s
/// object model. Fundamentally, items in the stored array are implicitly
/// creatable objects.
///   - if we want numpy to pre-allocate the memory and pass it in and then
///     much later interpret it as the storage used by this class, we will need
///     to use std::start_lifetime_as.
///   - alternatively, we should be able to use numpy's and cpython's C API to
///     allocate data in C++ (or allocate it from Python and then call
///     placement new) and then when it comes time to remove the last reference
///     to the data, we use a callback to delete the referenec. Here is a
///     stackoverflow link that touches on some of the machinery you would use
///     in a simplified situation
///     https://stackoverflow.com/questions/6811749/how-to-register-a-destructor-for-a-c-allocated-numpy-array
///
///
/// sure we get alignment and offsets exactly right)
///   * there's an added challenge in C++ to ensure that we handle object
///     creation properly (e.g. using placement-new or std::start_lifetime_as)
template <int N_i64, int N_f64>
class StatDataView {
  using ImplProp = detail::ImplProps_<N_i64, N_f64>;
  using element_type = typename ImplProp::element_type;

public:
  StatDataView() = default;

  // destructor
  ~StatDataView() {
    if (data_ != nullptr) delete[] data_;
  }

  StatDataView(std::size_t n_registers, std::size_t n_i64, std::size_t n_f64)
      : n_registers_(n_registers), n_i64_(n_i64), n_f64_(n_f64) {
    require(n_registers > 0, "this constructor requires n_registers > 0");

    if (N_i64 == -1) {
      require(n_i64 > 0, "n_i64 must be positive");
    } else {
      require(n_i64 == N_i64, "n_i64 doesn't match the template parameter");
    }

    if (N_f64 == -1) {
      require(n_f64 > 0, "n_f64 must be positive");
    } else {
      require(n_f64 == N_f64, "n_f64 doesn't match the template parameter");
    }

    // compute the total number of elements
    std::size_t len;
    if ((N_i64 == -1) && (N_f64 == -1)) {
      error("this branch shouldn't be reachable");
    } else if (N_i64 == -1) {
      len = std::size_t(n_registers) * std::size_t(n_i64);
    } else if (N_f64 == -1) {
      len = std::size_t(n_registers) * std::size_t(n_f64);
    } else {
      len = std::size_t(n_registers);
    }
    data_ = new element_type[len];
    n_bytes_ = sizeof(element_type) * len;

    this->zero_fill();
  }

  // delete copy operations
  StatDataView(const StatDataView<N_i64, N_f64>&) = delete;
  StatDataView<N_i64, N_f64>& operator=(const StatDataView<N_i64, N_f64>&) =
      delete;

  /// move constructor
  StatDataView(StatDataView<N_i64, N_f64>&& o)
      : data_(std::exchange(o.data_, nullptr)),
        n_registers_(std::exchange(o.n_registers_, 0)),
        n_i64_(std::exchange(o.n_i64_, 0)),
        n_f64_(std::exchange(o.n_f64_, 0)),
        n_bytes_(std::exchange(o.n_bytes_, 0)) {}

  /// move assignment
  StatDataView<N_i64, N_f64>& operator=(StatDataView<N_i64, N_f64>&& o) {
    this->data_ = std::exchange(o.data_, nullptr);
    this->n_registers_ = std::exchange(o.n_registers_, 0);
    this->n_i64_ = std::exchange(o.n_i64_, 0);
    this->n_f64_ = std::exchange(o.n_f64_, 0);
    this->n_bytes_ = std::exchange(o.n_bytes_, 0);

    return *this;
  }

  /// return a deepcopy of ``this``
  StatDataView<N_i64, N_f64> clone() const noexcept {
    // it would be faster to memcpy
    StatDataView<N_i64, N_f64> out(n_registers_, n_i64_, n_f64_);
    out.inplace_add(*this);
    return out;
  }

  /// retrieve the specified i64 value
  ///
  /// register_index is associated with the spatial separation bin
  FORCE_INLINE std::int64_t& get_i64(std::size_t register_index,
                                     std::size_t idx) const noexcept {
    // require((0 <= register_index) && (register_index < n_registers_),
    //         "invalid register_index");
    // require((0 <= idx) && (idx < n_i64_), "invalid idx");
    if constexpr (N_i64 == -1) {
      return data_[idx + register_index * n_i64_];
    } else if constexpr (N_i64 == 0) {
      error("this function should not be executed");
    } else {
      return data_[register_index].i64_vals[idx];
    }
  }

  /// retrieve the specified f64 value
  ///
  /// register_index is associated with the spatial separation bin
  FORCE_INLINE double& get_f64(std::size_t register_index,
                               std::size_t idx) const noexcept {
    // require((0 <= register_index) && (register_index < n_registers_),
    //         "invalid register_index");
    // require((0 <= idx) && (idx < n_f64_), "invalid idx");
    if constexpr (N_f64 == -1) {
      return data_[idx + register_index * n_f64_];
    } else if constexpr (N_f64 == 0) {
      error("this function should not be executed");
    } else {
      return data_[register_index].f64_vals[idx];
    }
  }

  template <typename T>
  FORCE_INLINE T& get(std::size_t register_index,
                      std::size_t idx) const noexcept {
    if constexpr (std::is_same_v<T, std::int64_t>) {
      return get_i64(register_index, idx);
    } else {
      return get_f64(register_index, idx);
    }
  }

  /// return a pointer to the f64 data for the given register
  ///
  /// in the future, we may choose to get rid of this so that we can be more
  /// agnositic about the underlying data layout
  double* get_f64_register_ptr(std::size_t register_idx,
                               std::size_t f64_offset = 0) const {
    if constexpr (N_f64 == -1) {
      return data_ + (f64_offset + register_idx + n_f64_);
    } else if constexpr (N_f64 == 0) {
      error("this function should not be executed");
    } else {
      return data_[register_idx].f64_vals + f64_offset;
    }
  }

  /// overwrite all entries in `this` with values of 0
  void zero_fill() const {
    static_assert(std::is_trivially_copyable_v<element_type>);
    if (!is_empty()) std::memset(data_, 0, n_bytes_);
  }

  /// Updates the values of `*this` to include the values from `other`
  void inplace_add(const StatDataView<N_i64, N_f64>& other) const {
    require((data_ != nullptr) && (other.data_ != nullptr) &&
                (n_registers_ == other.n_registers_) &&
                (n_i64_ == other.n_i64_) && (n_f64_ == other.n_f64_),
            "the contents of other can't be added to this");

    if ((N_i64 == -1) && (N_f64 == -1)) {
      error(
          "this branch shouldn't be reachable. Somehow we have variable "
          "numbers of ints and floats");
    } else if constexpr (N_i64 == -1) {
      const std::size_t stop = n_registers_ * n_i64_;
      for (std::size_t i = 0; i < stop; i++) {
        data_[i] += other.data_[i];
      }
    } else if constexpr (N_f64 == -1) {
      const std::size_t stop = n_registers_ * n_f64_;
      for (std::size_t i = 0; i < stop; i++) {
        data_[i] += other.data_[i];
      }
    } else {
      for (std::size_t register_idx = 0; register_idx < n_registers_;
           register_idx++) {
        for (std::size_t i = 0; i < n_i64_; i++) {
          this->get_i64(register_idx, i) += other.get_i64(register_idx, i);
        }

        for (std::size_t i = 0; i < n_f64_; i++) {
          this->get_f64(register_idx, i) += other.get_f64(register_idx, i);
        }
      }
    }
  }

  /// overwrites the contents of ``this`` in register_idx with the values
  /// stored in the corresponding register of other
  void overwrite_register_from_other(const StatDataView<N_i64, N_f64>& other,
                                     std::size_t register_idx) const {
    for (std::size_t i = 0; i < n_i64_; i++) {
      this->get_i64(register_idx, i) = other.get_i64(register_idx, i);
    }

    for (std::size_t i = 0; i < n_f64_; i++) {
      this->get_f64(register_idx, i) = other.get_f64(register_idx, i);
    }
  }

  bool is_empty() const { return data_ == nullptr; }

  std::size_t num_registers() const { return n_registers_; }
  std::size_t num_i64() const { return n_i64_; }
  std::size_t num_f64() const { return n_f64_; }

private:
  /// number of elements
  element_type* data_ = nullptr;  // uses default member initializer
  /// number of registers
  std::size_t n_registers_ = 0;
  /// number of integers in a register
  std::size_t n_i64_ = 0;
  /// number of doubles in a register
  std::size_t n_f64_ = 0;
  /// total number of bytes that were allocated
  std::size_t n_bytes_ = 0;
};

#endif /* STATDATAVIEW_HPP */
