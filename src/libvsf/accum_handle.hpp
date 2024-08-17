// Define the C interface for creating a handle for accumulator collections
//
// This will support transfering data to and from the accumulator collection.
// This interface is primarily being created in anticipation of moving
// consolidation routines to the C++ library
//
// TODO: it would be REALLY nice to be able to directly and dynamically query
//       the required size of each output buffer. (And to be able to query the
//       meaning of each value)

#include "vsf.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/// Allocates the specified AccumulatorCollection and returns a handle to it
///
/// @param[in]  stat_list Pointer to an array of 1 or more StatListItems that
///     provide details about the statistics that will be computed.
/// @param[in]  stat_list_len Specifies the number of entries in stat_list.
/// @param[in]  num_dist_bins The number of distance bins used in the
///     accumulator.
void* accumhandle_create(const StatListItem* stat_list, size_t stat_list_len,
                         size_t num_dist_bins);

/// Deallocates the AccumulatorCollection associated with the handle
void accumhandle_destroy(void* handle);

/// Saves the values stored in an Accumulator Collection to pre-allocated
/// external arrays
///
/// @param[in]  handle The previously allocated accumulator collection handle,
///     from which data is copied.
/// @param[out] out_flt_vals Preallocated arrays to hold the output floating
///     point values.
/// @param[out] out_i64_vals Preallocated array to hold the output int64_t
///     values.
void accumhandle_export_data(void* handle, double* out_flt_vals,
                             int64_t* out_i64_vals);

/// Restore the state of an Accumulator Collection from values stored in
/// external buffers
///
/// This is primarily intended to be passed arrays that had previously been
/// modified by ``accumhandle_export_data``
///
/// @param[in,out] handle The previously allocated accumulator collection
///     handle, which will be modified
/// @param[in]     in_flt_vals Array of floating point values.
/// @param[in]     in_i64_vals Array of int64_t values.
void accumhandle_restore(void* handle, const double* in_flt_vals,
                         const int64_t* in_i64_vals);

/// Updates `handle_primary` with the consolidated values of itself with
/// `handle_secondary`
void accumhandle_consolidate_into_primary(void* handle_primary,
                                          void* handle_secondary);

#ifdef __cplusplus
}
#endif
