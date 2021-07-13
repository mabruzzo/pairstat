CC = g++
CFLAGS = -g -O2 -Wall -fPIC

LIBS=-lm

all: libvsf.so

libvsf.so: src/vsf.hpp src/vsf.cpp src/accumulators.hpp
	$(CC) $(CFLAGS) $(LIBS) -shared src/vsf.cpp -o libvsf.so

clean: libvsf.so
	rm libvsf.so
