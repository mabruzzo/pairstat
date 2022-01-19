CC = g++

# -fno-math-errno lets compilers inline the sqrt command (see comments of to
# this SO answer) https://stackoverflow.com/a/54642811/4538758
CFLAGS = -g -O2 -Wall -fPIC -fno-math-errno -fopenmp --std=c++17


LIBS=-lm

DEPS = src/vsf.hpp src/vsf.cpp \
src/accum_handle.hpp src/accum_handle.cpp \
src/accum_col_variant.hpp \
src/accumulators.hpp \
src/compound_accumulator.hpp \
src/partition.hpp \
src/utils.hpp

.PHONY: clean clean_cython clean_all

all: libvsf.so


libvsf.so: $(DEPS)
	$(CC) $(CFLAGS) $(LIBS) -shared src/accum_handle.cpp src/vsf.cpp -o src/libvsf.so

clean:
	rm -f src/libvsf.so

clean_cython:
	rm -rf build
	rm -rf pyvsf.egg-info
	rm -rf pyvsf/*.cpp
	rm -rf pyvsf/*.so

clean_all: clean clean_cython
