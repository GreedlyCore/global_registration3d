set(STD_REG_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

pybind11_add_module(std_solver
  ${STD_REG_SRC_DIR}/STDesc.cpp
  std_bind.cpp
)

target_include_directories(std_solver PRIVATE
  ${STD_REG_SRC_DIR}/../include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

target_compile_definitions(std_solver PRIVATE ${PCL_DEFINITIONS})
target_link_directories(std_solver PRIVATE ${PCL_LIBRARY_DIRS})

target_link_libraries(std_solver PRIVATE
  ${PCL_LIBRARIES}
  Ceres::ceres
)

set_target_properties(std_solver PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../../eval"
)