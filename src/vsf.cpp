#include <cmath>
#include <cstdio>
#include <cstdint>

#include <algorithm>
#include <vector>


// the anonymous namespace informs the compiler that the contents are only used
// in the local compilation unit (facillitating more optimizations)
namespace{

  struct VarAccum {

    VarAccum() : count(0), mean(0.0), cur_M2(0.0) {}

    inline void add_entry(double val) noexcept{
      count++;
      double val_minus_last_mean = val - mean;
      mean += (val_minus_last_mean)/count;
      double val_minus_cur_mean = val - mean;
      cur_M2 += val_minus_last_mean * val_minus_cur_mean;
    }

    // this gives the variance
    inline double get_variance() noexcept{
      if (count > 1){
        return cur_M2 / (count - 1);
      } else {
        return 0.0;
      }
    }

    // number of entries included (so far)
    int64_t count;
    // current mean
    double mean;
    // sum of differences from the current mean
    double cur_M2;

  };



  /// identify the index of the bin where x lies.
  ///
  /// @param x The value that is being queried
  /// @param bin_edges An array of monotonically increasing bin edges. This must
  ///    have ``nbins + 1`` entries. The ith bin includes the interval
  ///    ``bin_edges[i] <= x < bin_edges[i]``.
  /// @param nbins The number of bins. This is expected to be at least 1.
  ///
  /// @returns index The index that ``x`` belongs in. If ``x`` doesn't lie in any
  ///    bin, ``nbins`` is returned.
  ///
  /// @notes
  /// At the moment we are using a binary search algorithm. In the future, we
  /// might want to assess the significance of branch mispredictions.
  template<typename T>
    std::size_t identify_bin_index(T x, const T *bin_edges, std::size_t nbins)
  {
    if ((x < bin_edges[0]) || (x >= bin_edges[nbins])){
      return nbins;
    }
    const T* rslt = std::lower_bound(bin_edges, bin_edges+nbins, x);
    return std::distance(bin_edges, rslt);
  }


  std::vector<VarAccum> process_data(const double **pos_a, const double **vel_a,
                                     std::size_t len_a,
                                     const double **pos_b, const double **vel_b,
                                     std::size_t len_b,
                                     uint8_t ndim, const double *bin_edges,
                                     std::size_t nbins)
  {

    std::vector<VarAccum> accumulators(nbins);

    const double *x_a = pos_a[0];
    const double *y_a = pos_a[1];
    const double *z_a = pos_a[2];
    const double *vx_a = vel_a[0];
    const double *vy_a = vel_a[1];
    const double *vz_a = vel_a[2];

    const double *x_b = pos_b[0];
    const double *y_b = pos_b[1];
    const double *z_b = pos_b[2];
    const double *vx_b = vel_b[0];
    const double *vy_b = vel_b[1];
    const double *vz_b = vel_b[2];

    for (std::size_t i_a = 0; i_a < len_a; i_a++){
      for (std::size_t i_b = 0; i_b < len_b; i_b++){

        double dist = std::sqrt((x_a[i_a] - x_b[i_b])*(x_a[i_a] - x_b[i_b]) +
                                (y_a[i_a] - y_b[i_b])*(y_a[i_a] - y_b[i_b]) +
                                (z_a[i_a] - z_b[i_b])*(z_a[i_a] - z_b[i_b]));

        double abs_vdiff =
          std::sqrt((vx_a[i_a] - vx_b[i_b])*(vx_a[i_a] - vx_b[i_b]) +
                    (vy_a[i_a] - vy_b[i_b])*(vy_a[i_a] - vy_b[i_b]) +
                    (vz_a[i_a] - vz_b[i_b])*(vz_a[i_a] - vz_b[i_b]));

        std::size_t bin_ind = identify_bin_index(dist, bin_edges, nbins);
        if (bin_ind < nbins){
          accumulators[bin_ind].add_entry(abs_vdiff);
        }

      }
    }

    return accumulators;
  }

}


bool calc_vsf_props(const double **pos_a, const double **vel_a,
                    std::size_t len_a,
                    const double **pos_b, const double **vel_b,
                    std::size_t len_b,
                    uint8_t ndim, const double *bin_edges, std::size_t nbins,
                    double *out_vals, int64_t *out_counts)
{
  if (nbins == 0){
    return false;
  }

  if (ndim != 3){
    return false;
  }

  std::vector<VarAccum> accumulators = process_data(pos_a, vel_a, len_a,
                                                    pos_b, vel_b, len_b,
                                                    ndim,
                                                    bin_edges, nbins);

  for (std::size_t i = 0; i < nbins; i++){
    out_counts[i] = accumulators[i].count;
    out_vals[i] = accumulators[i].mean;
    out_vals[i + nbins] = accumulators[i].get_variance();
  }

  return true;
}


/*
int main(){


  double x[8] = { 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0};
  double y[8] = { 8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
  double z[8] = {16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0};


  double vx[8] = {24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0};
  double vy[8] = {32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  double vz[8] = {40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0};

  printf("THIS IS A TEST\n");

  return 0;
}
*/
