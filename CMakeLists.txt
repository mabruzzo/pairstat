# Prologue
# --------
# we require 3.21 at the recommendation of the cython-cmake python-package
#   https://github.com/scikit-build/cython-cmake
cmake_minimum_required(VERSION 3.21)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Dependencies
# ------------

# find required python components
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

option(LIBVSF_NO_OPENMP
  "When true, we won't use OpenMP. Otherwise, we use it if available." OFF)
if (LIBVSF_NO_OPENMP)
  set(OpenMP_CXX_FOUND "OFF")
else()
  find_package(OpenMP OPTIONAL_COMPONENTS CXX)
endif()

# Actual Build Recipes
# --------------------

# first, declare the pure C++ part of the library

add_library(vsf_cpp OBJECT src/libvsf/accum_handle.cpp src/libvsf/vsf.cpp)
set_target_properties(vsf_cpp PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(vsf_cpp PRIVATE
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
target_include_directories(vsf_cpp PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/libvsf")
if (CMAKE_CXX_COMPILER_ID MATCHES "^GNU|Clang|AppleClang$")
  # -fno-math-errno lets compilers inline the sqrt command (see comments of to
  # this SO answer) https://stackoverflow.com/a/54642811/4538758
  target_compile_options(vsf_cpp PRIVATE "-fno-math-errno")
else()
  message(FATAL_ERROR "this was a problem!")
endif()

# read in cython utilities (provided by cython-cmake python-package)
include(UseCython)

# invoking include_directories is crude, but necessary for add_cython_target
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/libvsf")

cython_transpile(
  src/pairstat/_ArrayDict_cy.pyx
  LANGUAGE CXX
  CYTHON_ARGS -3
  OUTPUT_VARIABLE _ArrayDict_cxx
)
python_add_library(_ArrayDict_cy MODULE ${_ArrayDict_cxx} WITH_SOABI)

cython_transpile(
  src/pairstat/_kernels_cy.pyx
  LANGUAGE CXX
  CYTHON_ARGS -3
  OUTPUT_VARIABLE _kernels_cxx
)
python_add_library(_kernels_cy MODULE ${_kernels_cxx} WITH_SOABI)
target_link_libraries(_kernels_cy PRIVATE vsf_cpp)

cython_transpile(
  src/pairstat/_partition_cy.pyx
  LANGUAGE CXX
  CYTHON_ARGS -3
  OUTPUT_VARIABLE _partition_cxx
)
python_add_library(_partition_cy MODULE ${_partition_cxx} WITH_SOABI)

# Installation Logic
# ------------------
install(TARGETS _ArrayDict_cy _kernels_cy _partition_cy
  DESTINATION ${SKBUILD_PROJECT_NAME})
