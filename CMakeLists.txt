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

# next, declare the cython packages

include("cmake/use_cython.cmake")

# invoking include_directories is crude, but necessary for add_cython_target
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/libvsf")

add_cython_target(_ArrayDict_cythonization "src/pyvsf/_ArrayDict_cy.pyx"
  CXX PY3 OUTPUT_VAR _ArrayDict_cxx_file)
python_add_library(_ArrayDict_cy MODULE ${_ArrayDict_cxx_file} WITH_SOABI)

add_cython_target(_kernels_cythonization "src/pyvsf/_kernels_cy.pyx"
  CXX PY3 OUTPUT_VAR _kernels_cxx_file)
python_add_library(_kernels_cy MODULE ${_kernels_cxx_file} WITH_SOABI)
target_link_libraries(_kernels_cy PRIVATE vsf_cpp)

add_cython_target(_partition_cythonization "src/pyvsf/_partition_cy.pyx"
  CXX PY3 OUTPUT_VAR _partition_cxx_file)
python_add_library(_partition_cy MODULE ${_partition_cxx_file} WITH_SOABI)

# Installation Logic
# ------------------
install(TARGETS _ArrayDict_cy _kernels_cy _partition_cy
  DESTINATION ${SKBUILD_PROJECT_NAME})
