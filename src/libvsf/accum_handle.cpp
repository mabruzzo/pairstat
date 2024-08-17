#include "accum_handle.hpp"

#include <cstdint>      // std::int64_t
#include <type_traits>  // std::decay

#include "accum_col_variant.hpp"

void* accumhandle_create(const StatListItem* stat_list,
                         std::size_t stat_list_len, std::size_t num_dist_bins) {
  if (stat_list_len != 1) {
    // it currently doesn't make any sense to try to work with
    // CompoundAccumCollection...
    // it doesn't define everything necessary to be useful
    error(
        "This function currently only expects a single stat_list item to be "
        "passed.");
  }

  // this is very inefficient, but we don't have a ton of options if we want
  // to avoid repeating a lot of code
  AccumColVariant tmp =
      build_accum_collection(stat_list, stat_list_len, num_dist_bins);
  AccumColVariant* out = new AccumColVariant(tmp);
  return static_cast<void*>(out);
}

void accumhandle_destroy(void* handle) {
  AccumColVariant* ptr = static_cast<AccumColVariant*>(handle);
  delete ptr;
}

void accumhandle_export_data(void* handle, double* out_flt_vals,
                             int64_t* out_i64_vals) {
  AccumColVariant* ptr = static_cast<AccumColVariant*>(handle);
  std::visit([=](auto& accum) { accum.copy_flt_vals(out_flt_vals); }, *ptr);
  std::visit([=](auto& accum) { accum.copy_i64_vals(out_i64_vals); }, *ptr);
}

void accumhandle_restore(void* handle, const double* in_flt_vals,
                         const int64_t* in_i64_vals) {
  AccumColVariant* ptr = static_cast<AccumColVariant*>(handle);
  std::visit([=](auto& accum) { accum.import_vals(in_flt_vals); }, *ptr);
  std::visit([=](auto& accum) { accum.import_vals(in_i64_vals); }, *ptr);
}

void accumhandle_consolidate_into_primary(void* handle_primary,
                                          void* handle_secondary) {
  AccumColVariant* primary_ptr = static_cast<AccumColVariant*>(handle_primary);
  AccumColVariant* secondary_ptr =
      static_cast<AccumColVariant*>(handle_secondary);

  std::visit(
      [=](auto& accum) {
        using T = std::decay_t<decltype(accum)>;
        if (std::holds_alternative<T>(*secondary_ptr)) {
          accum.consolidate_with_other(std::get<T>(*secondary_ptr));
        } else {
          error("the arguments don't hold the same types of accumulators");
        }
      },
      *primary_ptr);
}
