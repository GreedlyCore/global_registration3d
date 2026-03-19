cmake_minimum_required(VERSION 3.16)
project(adaptive_geometric_bootstrapping LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters kdtree)
find_package(OpenMP REQUIRED)

add_executable(adaptive_bootstrap main.cpp)

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
