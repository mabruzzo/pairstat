#include "accum_handle.hpp"

#include <cstdint>      // std::int64_t
#include <type_traits>  // std::decay

#include "accum_col_variant.hpp"

void* accumhandle_create(const StatListItem* stat_list,
                         std::size_t stat_list_len, std::size_t num_dist_bins) {
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
  std::visit([=](auto& accum) { accum.copy_vals(out_flt_vals); }, *ptr);
  std::visit([=](auto& accum) { accum.copy_vals(out_i64_vals); }, *ptr);
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

void accumhandle_add_entries(void* handle, int purge_everything_first,
                             std::size_t spatial_bin_index,
                             std::size_t num_entries, double* values,
                             double* weights) {
  AccumColVariant* ptr = static_cast<AccumColVariant*>(handle);
  if ((purge_everything_first < 0) || (purge_everything_first > 1)) {
    error("purge_everything_first must be 0 or 1");
  } else if ((num_entries > 0) && (values == nullptr)) {
    error("values can't be a nullptr when num_entries is positive");
  }

  std::visit(
      [=](auto& accum) {
        using T = std::decay_t<decltype(accum)>;
        // more argument checks
        if (spatial_bin_index >= accum.n_spatial_bins()) {
          error("spatial_bin_index is too big");
        } else if (T::requires_weight && (num_entries > 0) &&
                   (weights == nullptr)) {
          error("weights arg can't be a nullptr for the current handle.");
        }

        if (purge_everything_first) {
          accum.purge();
        }

        for (size_t i = 0; i < num_entries; i++) {
          if constexpr (T::requires_weight) {
            accum.add_entry(spatial_bin_index, values[i], weights[i]);
          } else {
            accum.add_entry(spatial_bin_index, values[i]);
          }
        }
      },
      *ptr);
}

void accumhandle_postprocess(void* handle) {
  AccumColVariant* ptr = static_cast<AccumColVariant*>(handle);
  std::visit([=](auto& accum) { accum.postprocess(); }, *ptr);
}

static PropDescr get_descr_(void* handle, int i) {
  AccumColVariant* ptr = static_cast<AccumColVariant*>(handle);
  PropDescr descr;
  std::visit([&](auto& accum) { descr = accum.try_get_prop_descr(i); }, *ptr);
  return descr;
}

const char* accumhandle_prop_name(void* handle, int i) {
  return get_descr_(handle, i).name;
}

int accumhandle_prop_isdouble(void* handle, int i) {
  return get_descr_(handle, i).is_f64 ? 1 : 0;
}

int accumhandle_prop_count(void* handle, int i) {
  return get_descr_(handle, i).count;
}
