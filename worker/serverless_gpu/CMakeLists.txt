cmake_minimum_required(VERSION 3.13)

project(svgpu-manager C CXX)

add_executable(svgpu-manager
  manager.cpp
  ${manager_service_proto_srcs}
  ${manager_service_ava_srcs}
)
target_link_libraries(svgpu-manager
  ${manager_service_libraries}
  ${absl_flags_libraries}
)
set_target_properties(svgpu-manager
  PROPERTIES OUTPUT_NAME manager
)
add_compile_options(-Wall -Wextra -pedantic -Werror -Wno-write-strings)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/manager
  TYPE BIN
  RENAME svgpu_manager
)
