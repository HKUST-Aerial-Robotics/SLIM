
cmake_minimum_required(VERSION 3.0)
project(pmc_slim CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(pmc_slim STATIC
  pmc_utils.cpp
  pmc_graph.cpp
  pmc_clique_utils.cpp
  pmc_heu.cpp
  pmc_maxclique.cpp
  pmcx_maxclique.cpp
  pmcx_maxclique_basic.cpp
  pmc_cores.cpp
)

