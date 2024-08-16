clean:
	rm -f src/libvsf.so

clean_cython:
	rm -rf build
	rm -rf pyvsf.egg-info
	rm -rf pyvsf/*.cpp
	rm -rf pyvsf/*.so

clean_all: clean clean_cython
