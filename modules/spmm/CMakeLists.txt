cmake_minimum_required(VERSION 3.1)

project(spmm LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -fPIC")
endif()

find_package(Eigen REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(CUDAToolkit REQUIRED)

include(GNUInstallDirs)

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Debug options
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -O0 -g -lineinfo")

add_library(spmm STATIC
  spmm.cpp
)

set_property(TARGET spmm PROPERTY CUDA_ARCHITECTURES OFF)
target_include_directories(spmm
    PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(spmm
  PUBLIC
  CUDA::cudart 
  CUDA::cusparse 
  CUDA::cublas
  CUDA::cusparse
  CUDA::cusolver
)