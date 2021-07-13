#ifndef ACCUMULATORS_H
#define ACCUMULATORS_H

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

#endif /* ACCUMULATORS_H */
