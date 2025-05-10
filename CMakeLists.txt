cmake_minimum_required(VERSION 3.5)
project(SLIM LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -fPIC")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

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

find_package(Eigen REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Boost REQUIRED COMPONENTS regex thread system filesystem)
find_package(Boost REQUIRED COMPONENTS graph filesystem)

add_definitions(-DBOOST_NO_CXX11_SCOPED_ENUMS)

add_subdirectory(thirdparty/pmc)
add_subdirectory(modules/spmm)
add_subdirectory(backward-cpp)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/include
  modules
  modules/factor
  modules/dbscan
  thirdparty
  thirdparty/pmc
)

add_library(SLIM
  modules/vector_map.cpp
  modules/viewer.cpp
  modules/frame.cpp
  modules/landmark.cpp
  modules/voxel.cpp
  modules/pcm_solver.cpp
  modules/extractor.cpp
  modules/scene_graph.cpp
  modules/multi_map_merger.cpp
  modules/clipper.cpp
  modules/locator.cpp
  modules/nfr_solver.cpp
  modules/json.cpp
  modules/observation.cpp
  modules/utility.cpp
  modules/factor/marginalization_factor.cpp
  modules/factor/laser_edge_factor.cpp
  modules/factor/laser_surf_factor.cpp
  modules/factor/relative_pose_factor.cpp
  modules/factor/prior_pose_factor.cpp
  modules/factor/local_param.cpp
)

target_link_libraries(SLIM
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Pangolin_LIBRARY}
  ${Boost_LIBRARIES}
  ${Boost_LIBRARY_DIRS}
  spmm
  OpenMP::OpenMP_CXX
  pmc_slim
)

add_executable(offline_mapping test/offline_mapping.cpp ${BACKWARD_ENABLE})
target_link_libraries(offline_mapping SLIM )
add_backward(offline_mapping)

add_executable(cloud_visualizer test/cloud_visualizer.cpp)
target_link_libraries(cloud_visualizer SLIM )

add_executable(map_merge test/map_merge.cpp ${BACKWARD_ENABLE})
target_link_libraries(map_merge SLIM )
add_backward(map_merge)

add_executable(map_visualizer test/map_visualizer.cpp ${BACKWARD_ENABLE})
target_link_libraries(map_visualizer SLIM )
add_backward(map_visualizer)

add_executable(relocalization test/relocalization.cpp ${BACKWARD_ENABLE})
target_link_libraries(relocalization SLIM )
add_backward(relocalization)