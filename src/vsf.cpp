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

#include "accumulators.hpp"
#include "compound_accumulator.hpp"
#include "accum_col_variant.hpp"


// the anonymous namespace informs the compiler that the contents are only used
// in the local compilation unit (facillitating more optimizations)
namespace{

  FORCE_INLINE double calc_dist_sqr(double x0, double x1,
                                    double y0, double y1,
                                    double z0, double z1) noexcept{
    double dx = x0 - x1;
    double dy = y0 - y1;
    double dz = z0 - z1;
    return dx*dx + dy*dy + dz*dz;
  }

  struct dist_rslt{ double dist_sqr; double abs_vdiff; };

  FORCE_INLINE dist_rslt calc_dist_rslt(double x_a, double y_a, double z_a,
                                        double vx_a, double vy_a, double vz_a,
                                        const double* pos_b,
                                        const double* vel_b,
                                        std::size_t i_b,
                                        std::size_t n_points) noexcept
  {
    const double x_b = pos_b[i_b];
    const double y_b = pos_b[i_b + n_points];
    const double z_b = pos_b[i_b + 2*n_points];

    const double vx_b = vel_b[i_b];
    const double vy_b = vel_b[i_b + n_points];
    const double vz_b = vel_b[i_b + 2*n_points];

    double dist_sqr = calc_dist_sqr(x_a, x_b,
                                    y_a, y_b,
                                    z_a, z_b);
    double abs_vdiff = std::sqrt(calc_dist_sqr(vx_a, vx_b,
                                               vy_a, vy_b,
                                               vz_a, vz_b));
    return {dist_sqr, abs_vdiff};
  };

  

  template<class AccumCollection, bool duplicated_points>
  void process_data(const PointProps points_a,
                    const PointProps points_b,
                    const double *dist_sqr_bin_edges,
                    std::size_t nbins,
                    AccumCollection& accumulators)
  {

    // this assumes 3D

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

      const double x_a = pos_a[i_a];
      const double y_a = pos_a[i_a + n_points_a];
      const double z_a = pos_a[i_a + 2*n_points_a];

      const double vx_a = vel_a[i_a];
      const double vy_a = vel_a[i_a + n_points_a];
      const double vz_a = vel_a[i_a + 2*n_points_a];

      for (std::size_t i_b = i_b_start; i_b < n_points_b; i_b++){

        dist_rslt tmp = calc_dist_rslt(x_a, y_a, z_a,
                                       vx_a, vy_a, vz_a,
                                       pos_b, vel_b, i_b, n_points_b);

	std::size_t bin_ind = identify_bin_index(tmp.dist_sqr,
                                                 dist_sqr_bin_edges,
                                                 nbins);
	if (bin_ind < nbins){
          accumulators.add_entry(bin_ind, tmp.abs_vdiff);
	}
      }
    }
  }

  template<typename AccumCollection>
  void calc_vsf_props_helper_(const PointProps points_a,
			      const PointProps points_b,
			      const double *dist_sqr_bin_edges,
                              std::size_t nbins,
                              AccumCollection& accumulators,
			      double *out_flt_vals, int64_t *out_i64_vals,
			      bool duplicated_points){

    if (duplicated_points){
      process_data<AccumCollection, true>(points_a, points_b,
                                          dist_sqr_bin_edges, nbins, 
                                          accumulators);
    } else {
      process_data<AccumCollection, false>(points_a, points_b,
                                           dist_sqr_bin_edges, nbins,
                                           accumulators);
    }

    accumulators.copy_flt_vals(out_flt_vals);
    accumulators.copy_i64_vals(out_i64_vals);

  }
}


bool calc_vsf_props(const PointProps points_a, const PointProps points_b,
		    const StatListItem* stat_list, std::size_t stat_list_len,
                    const double *bin_edges, std::size_t nbins,
                    double *out_flt_vals, int64_t *out_i64_vals) noexcept
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

  // recompute the bin edges so that they are stored as squared distances
  std::vector<double> dist_sqr_bin_edges_vec(nbins+1);
  for (std::size_t i=0; i < (nbins+1); i++){
    if (bin_edges[i] < 0){
      // It doesn't really matter how we handle negative bin edges (since
      // distances are non-negative), as long as dist_sqr_bin_edges
      // monotonically increases.
      dist_sqr_bin_edges_vec[i] = bin_edges[i];
    } else {
      dist_sqr_bin_edges_vec[i] = bin_edges[i]*bin_edges[i];
    }
  }

  // construct accumulators (they're stored in a std::variant for convenience)
  AccumColVariant accumulators = build_accum_collection(stat_list,
                                                        stat_list_len,
                                                        nbins);

  // now actually use the accumulators to compute that statistics
  auto func = [=](auto& accumulators)
    {
      calc_vsf_props_helper_(points_a, my_points_b,
                             dist_sqr_bin_edges_vec.data(), nbins,
                             accumulators,
                             out_flt_vals, out_i64_vals,
                             duplicated_points);
    };

  std::visit(func, accumulators);

  return true;
}
