CC = g++

# -fno-math-errno lets compilers inline the sqrt command (see comments of to
# this SO answer) https://stackoverflow.com/a/54642811/4538758
CFLAGS = -g -O2 -Wall -fPIC -fno-math-errno --std=c++17

LIBS=-lm

all: libvsf.so

libvsf.so: src/vsf.hpp src/vsf.cpp src/accumulators.hpp src/compound_accumulator.hpp src/accum_col_variant.hpp
	$(CC) $(CFLAGS) $(LIBS) -shared src/vsf.cpp -o src/libvsf.so

clean: src/libvsf.so
	rm src/libvsf.so
