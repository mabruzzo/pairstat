#ifndef ACCUMULATORS_H
#define ACCUMULATORS_H

#include <utility> // std::pair

void error(const char* message){
  if (message == nullptr){
    printf("ERROR\n");
  } else {
    printf("ERROR: %s\n", message);
  }
  exit(1);
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

struct MeanAccum{

public: // interface
  static std::vector<std::string> flt_val_names() noexcept{
    return {"mean"};
  }

  double get_flt_val(std::size_t i) const noexcept{
    if (i == 0){
      return mean;
    } else {
      error("MeanAccum only has 1 float_val");
      return 0.0; // this prevents compiler complaints
    }
  }

  MeanAccum() : count(0), mean(0.0) {}

  inline void add_entry(double val) noexcept{
    count++;
    double val_minus_last_mean = val - mean;
    mean += (val_minus_last_mean)/count;
  }

public: // attributes
  // number of entries included (so far)
  int64_t count;
  // current mean
  double mean;
};


struct VarAccum {

  public: // interface
  static std::vector<std::string> flt_val_names() noexcept{
    return {"mean", "variance"};
  }

  double get_flt_val(std::size_t i) const noexcept{
    if (i == 0){
      return mean;
    } else if (i == 1){
      return (count > 1) ? cur_M2 / (count - 1) : 0.0;
    } else {
      error("VarAccum only has 2 float_vals");
      return 0.0; // this prevents compiler complaints
    }
  }

  VarAccum() : count(0), mean(0.0), cur_M2(0.0) {}

  inline void add_entry(double val) noexcept{
    count++;
    double val_minus_last_mean = val - mean;
    mean += (val_minus_last_mean)/count;
    double val_minus_cur_mean = val - mean;
    cur_M2 += val_minus_last_mean * val_minus_cur_mean;
  }

public: // attributes
  // number of entries included (so far)
  int64_t count;
  // current mean
  double mean;
  // sum of differences from the current mean
  double cur_M2;

};



template<typename Accum>
class ScalarAccumCollection{

public:

  ScalarAccumCollection() noexcept : accum_list_() {}

  ScalarAccumCollection(std::size_t n_spatial_bins, void * other_arg) noexcept
    : accum_list_(n_spatial_bins)
  {
    if (n_spatial_bins == 0) { error("n_spatial_bins must be positive"); }
    if (other_arg != nullptr) { error("other_arg must be nullptr"); }
  }

  inline void add_entry(std::size_t spatial_bin_index, double val) noexcept{
    accum_list_[spatial_bin_index].add_entry(val);
  }

  /// Return the Floating Point Value Properties
  ///
  /// This is a vector holding pairs of the value name and the number of value
  /// entries per spatial bin.
  ///
  /// For Scalar Accumulators, each value only stores 1 entry per spatial bin
  static std::vector<std::pair<std::string,std::size_t>> flt_val_props()
    noexcept
  {
    std::vector<std::string> flt_val_names = Accum::flt_val_names();

    std::vector<std::pair<std::string,std::size_t>> out;
    for (std::size_t i = 0; i < flt_val_names.size(); i++){
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
  static std::vector<std::pair<std::string,std::size_t>> i64_val_props()
    noexcept
  { return {{"count", 1}}; }

  void copy_flt_vals(double *out_vals) noexcept {
    const std::size_t num_flt_vals = Accum::flt_val_names().size();
    const std::size_t n_bins = accum_list_.size();

    for (std::size_t i = 0; i < n_bins; i++){
      for (std::size_t j = 0; j < num_flt_vals; j++){
	out_vals[i + j*n_bins] = accum_list_[i].get_flt_val(j);
      }
    }
  }

  void copy_i64_vals(int64_t *out_vals) noexcept {
    for (std::size_t i = 0; i < accum_list_.size(); i++){
      out_vals[i] = accum_list_[i].count;
    }
  }

  std::vector<Accum> get_accum_vector() const noexcept {
    return accum_list_;
  }

private:
  std::vector<Accum> accum_list_;

};

#endif /* ACCUMULATORS_H */
