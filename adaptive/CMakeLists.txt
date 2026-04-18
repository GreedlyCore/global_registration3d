cmake_minimum_required(VERSION 3.16)
project(adaptive_geometric_bootstrapping LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

find_package(MPI QUIET COMPONENTS C)
if(NOT TARGET MPI::MPI_C)
    add_library(MPI::MPI_C INTERFACE IMPORTED)
endif()

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters kdtree io)
find_package(OpenMP REQUIRED)

add_library(adaptive_core OBJECT adaptive_lib.cpp)

target_link_libraries(adaptive_core
    PUBLIC
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        ${PCL_LIBRARIES}
)

target_include_directories(adaptive_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PCL_INCLUDE_DIRS}
)

target_compile_definitions(adaptive_core
    PRIVATE
        ${PCL_DEFINITIONS}
)

option(BUILD_DEMO "Build adaptive demo executable" ON)
if(BUILD_DEMO)
    add_executable(adaptive_bootstrap main.cpp $<TARGET_OBJECTS:adaptive_core>)

    target_link_libraries(adaptive_bootstrap
        PRIVATE
            Eigen3::Eigen
            OpenMP::OpenMP_CXX
            ${PCL_LIBRARIES}
    )

    target_include_directories(adaptive_bootstrap
        PRIVATE
            ${PCL_INCLUDE_DIRS}
    )

    target_compile_definitions(adaptive_bootstrap
        PRIVATE
            ${PCL_DEFINITIONS}
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(adaptive_bootstrap PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON)
    find_package(pybind11 QUIET CONFIG)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND python3 -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE pybind11_CMD_RESULT
        )
        if(NOT pybind11_CMD_RESULT EQUAL 0)
            message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
        endif()
        find_package(pybind11 REQUIRED CONFIG PATHS ${pybind11_DIR})
    endif()

    pybind11_add_module(adaptive_bootstrap_py adaptive_bind.cpp $<TARGET_OBJECTS:adaptive_core>)

    target_link_libraries(adaptive_bootstrap_py
        PRIVATE
            Eigen3::Eigen
            OpenMP::OpenMP_CXX
            ${PCL_LIBRARIES}
    )

    target_include_directories(adaptive_bootstrap_py
        PRIVATE
            ${PCL_INCLUDE_DIRS}
    )

    target_compile_definitions(adaptive_bootstrap_py
        PRIVATE
            ${PCL_DEFINITIONS}
    )

    set_target_properties(adaptive_bootstrap_py PROPERTIES
        PREFIX ""
        OUTPUT_NAME "adaptive_bootstrap"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../eval"
    )
endif()
