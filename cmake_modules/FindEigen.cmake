# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This find_module finds Eigen3 and assigns variables to standard variable names
# This allows the module to work with ament_export_dependencies

include(FindPackageHandleStandardArgs)

# Use Eigen3Config.cmake to do most of the work
find_package(Eigen3 CONFIG QUIET)

if(Eigen3_FOUND AND NOT EIGEN3_FOUND)
  # Special case for Eigen 3.3.4 chocolately package
  find_package_handle_standard_args(
    Eigen3
    VERSION_VAR
      Eigen3_VERSION
    REQUIRED_VARS
      Eigen3_FOUND
  )
else()
  find_package_handle_standard_args(
    Eigen3
    VERSION_VAR
      EIGEN3_VERSION_STRING
    REQUIRED_VARS
      EIGEN3_FOUND
  )
endif()

# Set standard variable names
# https://cmake.org/cmake/help/v3.5/manual/cmake-developer.7.html#standard-variable-names
set(_standard_vars
  INCLUDE_DIRS
  DEFINITIONS
  ROOT_DIR)
foreach(_suffix ${_standard_vars})
  set(_wrong_var "EIGEN3_${_suffix}")
  set(_right_var "Eigen3_${_suffix}")
  if(NOT DEFINED ${_right_var} AND DEFINED ${_wrong_var})
    set(${_right_var} "${${_wrong_var}}")
  endif()
endforeach()