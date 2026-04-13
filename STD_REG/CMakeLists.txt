cmake_minimum_required(VERSION 3.22)
project(std_reg_solver)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

find_package(pybind11 QUIET CONFIG)
if(NOT pybind11_FOUND)
  if(NOT Python3_EXECUTABLE)
    set(Python3_EXECUTABLE python3)
  endif()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_CMD_RESULT
  )
  if(NOT pybind11_CMD_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 not found. Install it with pip install pybind11")
  endif()
  find_package(pybind11 REQUIRED CONFIG PATHS ${pybind11_DIR})
endif()

add_subdirectory(src/pybind)