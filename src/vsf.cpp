#include <cmath>
#include <cstdio>
#include <cstdint>

#include <algorithm>
#include <string>
#include <vector>

#include "vsf.hpp"

#if defined(__GNUC__)
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define FORCE_INLINE inline
#endif

// the anonymous namespace informs the compiler that the contents are only used
// in the local compilation unit (facillitating more optimizations)
namespace{

  // intentionally include this header file within the anonymous namespace
  #include "accumulators.hpp"


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
  template<typename T>
    std::size_t identify_bin_index(T x, const T *bin_edges, std::size_t nbins)
  {
    const T* bin_edges_end = bin_edges+nbins+1;
    const T* rslt = std::lower_bound(bin_edges, bin_edges_end, x);
    // rslt is a pointer to the first value that is "not less than" x
    std::size_t index_p_1 = std::distance(bin_edges, rslt);

    if (index_p_1 == 0 || index_p_1 == (nbins + 1)){
      return nbins;
    } else {
      return index_p_1 - 1;
    }
  }

  
   FORCE_INLINE double dist_sqr_3D(const double* arr1, std::size_t i1,
				   std::size_t stride1,
				   const double* arr2, std::size_t i2,
				   std::size_t stride2){
    double dx = arr1[i1] - arr2[i2];
    double dy = arr1[i1 + stride1] - arr2[i2 + stride2];
    double dz = arr1[i1 + 2*stride1] - arr2[i2 + 2*stride2];
    return dx*dx + dy*dy + dz*dz;
  }

   template<typename Accum, bool duplicated_points>
  std::vector<Accum> process_data(const PointProps points_a,
				  const PointProps points_b,
				  const double *bin_edges,
				  std::size_t nbins)
  {

    // this assumes 3D
    std::vector<Accum> accumulators(nbins);

    const std::size_t n_points_a = points_a.n_points;
    const double *pos_a = points_a.positions;
    const double *vel_a = points_a.velocities;

    const std::size_t n_points_b = points_b.n_points;
    const double *pos_b = points_b.positions;
    const double *vel_b = points_b.velocities;

    
    for (std::size_t i_a = 0; i_a < n_points_a; i_a++){
      // When duplicated_points is true, points_a is the same as points_b. In
      // that case, take some care to avoid duplicating pairs
      std::size_t i_b_start = (duplicated_points) ? i_a + 1 : 0;
      for (std::size_t i_b = i_b_start; i_b < n_points_b; i_b++){

	double dist = std::sqrt(dist_sqr_3D(pos_a, i_a, n_points_a,
					    pos_b, i_b, n_points_b));
	double abs_vdiff = std::sqrt(dist_sqr_3D(vel_a, i_a, n_points_a,
						 vel_b, i_b, n_points_b));
	std::size_t bin_ind = identify_bin_index(dist, bin_edges, nbins);
	if (bin_ind < nbins){
	  accumulators[bin_ind].add_entry(abs_vdiff);
	}
      }
    }

    return accumulators;
  }

  template<typename Accum>
  void calc_vsf_props_helper_(const PointProps points_a,
			      const PointProps points_b,
			      const double *bin_edges, std::size_t nbins,
			      double *out_vals, int64_t *out_counts,
			      bool duplicated_points){

    std::vector<Accum> accumulators;
    if (duplicated_points){
      accumulators = process_data<Accum, true>(points_a, points_b,
					       bin_edges, nbins);
    } else {
      accumulators = process_data<Accum, false>(points_a, points_b,
						bin_edges, nbins);
    }

    const std::size_t num_flt_vals = Accum::flt_val_names().size();

    for (std::size_t i = 0; i < nbins; i++){
      out_counts[i] = accumulators[i].count;
      for (std::size_t j = 0; j < num_flt_vals; j++){
	out_vals[i + j*nbins] = accumulators[i].get_flt_val(j);
      }
    }

  }
}


bool calc_vsf_props(const PointProps points_a,
		    const PointProps points_b,
		    const char* statistic, const double *bin_edges,
		    std::size_t nbins,
		    double *out_vals, int64_t *out_counts)
{
  const bool duplicated_points = ((points_b.positions == nullptr) &&
				  (points_b.velocities == nullptr));

  const PointProps my_points_b = (duplicated_points) ? points_a : points_b;

  if (nbins == 0){
    return false;
  } else if (points_a.n_spatial_dims != 3){
    return false;
  } else if (my_points_b.n_spatial_dims != 3){
    return false;
  } else if ((points_a.positions == nullptr) ||
	     (points_a.velocities == nullptr)) {
    return false;
  }

  std::string stat_str(statistic);

  if (stat_str == "mean"){
    calc_vsf_props_helper_<MeanAccum>(points_a, my_points_b, bin_edges, nbins,
				      out_vals, out_counts, duplicated_points);
  } else if (std::string(statistic) == "variance"){
    calc_vsf_props_helper_<VarAccum>(points_a, my_points_b, bin_edges, nbins,
				     out_vals, out_counts, duplicated_points);
  } else {
    return false;
  }

  return true;
}
