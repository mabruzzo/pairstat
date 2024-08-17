# Prologue
# --------
cmake_minimum_required(VERSION 3.16)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Dependencies
# ------------

# find required python components
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# try to check if the c++ compiler supports openmp
find_package(OpenMP OPTIONAL_COMPONENTS CXX)

# Actual Build Recipies
# ---------------------
include("cmake/use_cython.cmake")

# invoking include_directories is crude, but necessary for add_cython_target
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/libvsf")

add_cython_target(_ArrayDict_cythonization "src/pyvsf/_ArrayDict_cy.pyx"
  CXX PY3 OUTPUT_VAR _ArrayDict_cxx_file)
python_add_library(_ArrayDict_cy MODULE ${_ArrayDict_cxx_file} WITH_SOABI)

# todo:
# -> consider compiling src/vsf.cpp and arc/accum_handle.cpp into a separate
#    library
# -> try re-enabling -fno-errno for vsf.cpp and accum_handle.cpp
#    -fno-math-errno lets compilers inline the sqrt command (see comments of to
#    this SO answer) https://stackoverflow.com/a/54642811/4538758

add_cython_target(_kernels_cythonization "src/pyvsf/_kernels_cy.pyx"
  CXX PY3 OUTPUT_VAR _kernels_cxx_file)
python_add_library(_kernels_cy MODULE
  ${_kernels_cxx_file}
  src/libvsf/accum_handle.cpp src/libvsf/vsf.cpp
  WITH_SOABI)
if (OpenMP_CXX_FOUND)
  target_link_libraries(_kernels_cy PRIVATE OpenMP::OpenMP_CXX)
endif()

add_cython_target(_partition_cythonization "src/pyvsf/_partition_cy.pyx"
  CXX PY3 OUTPUT_VAR _partition_cxx_file)
python_add_library(_partition_cy MODULE ${_partition_cxx_file} WITH_SOABI)

# Installation Logic
# ------------------
install(TARGETS _ArrayDict_cy _kernels_cy _partition_cy
  DESTINATION ${SKBUILD_PROJECT_NAME})
