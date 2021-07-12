CC = g++
CFLAGS = -g -O2 -Wall -fPIC

LIBS=-lm

all: libvsf.so

libvsf.so: src/vsf.cpp
	$(CC) $(CFLAGS) $(LIBS) -shared src/vsf.cpp -o libvsf.so

clean: libvsf.so
	rm libvsf.so
