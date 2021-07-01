cmake_minimum_required(VERSION 3.13)

project(svgpu-manager C CXX)

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)

find_program(_PROTOBUF_PROTOC protoc)

#find_package(absl CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
set(_GRPC_GRPCPP gRPC::grpc++)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

#gen grpc stuff
get_filename_component(gpu_proto "pb/gpuserver.proto" ABSOLUTE)
get_filename_component(gpu_proto_path "${gpu_proto}" PATH)
get_filename_component(rm_proto "pb/resmngr.proto" ABSOLUTE)
get_filename_component(rm_proto_path "${rm_proto}" PATH)

# Generated sources
set(gpu_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/gpuserver.pb.cc")
set(gpu_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/gpuserver.pb.h")
set(gpu_grpc_srcs  "${CMAKE_CURRENT_BINARY_DIR}/gpuserver.grpc.pb.cc")
set(gpu_grpc_hdrs  "${CMAKE_CURRENT_BINARY_DIR}/gpuserver.grpc.pb.h")

set(rm_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/resmngr.pb.cc")
set(rm_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/resmngr.pb.h")
set(rm_grpc_srcs  "${CMAKE_CURRENT_BINARY_DIR}/resmngr.grpc.pb.cc")
set(rm_grpc_hdrs  "${CMAKE_CURRENT_BINARY_DIR}/resmngr.grpc.pb.h")

add_custom_command(
  OUTPUT "${gpu_proto_srcs}" "${gpu_proto_hdrs}" "${gpu_grpc_srcs}" "${gpu_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
    --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
    -I "${gpu_proto_path}"
    --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
    "${gpu_proto}"
  DEPENDS "${gpu_proto}")

add_custom_command(
  OUTPUT "${rm_proto_srcs}" "${rm_proto_hdrs}" "${rm_grpc_srcs}" "${rm_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
    --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
    -I "${rm_proto_path}"
    --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
    "${rm_proto}"
  DEPENDS "${rm_proto}")

include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories("/usr/local/cuda-10.1/include")

add_library(grpc_proto
  ${gpu_grpc_srcs}
  ${gpu_grpc_hdrs}
  ${gpu_proto_srcs}
  ${gpu_proto_hdrs}
  ${rm_proto_srcs}
  ${rm_proto_hdrs}
  ${rm_grpc_srcs}
  ${rm_grpc_hdrs})

target_link_libraries(grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

add_executable(svgpu-manager
  manager.cpp
  SVGPUManager.cpp
  ${manager_service_proto_srcs}
  ${manager_service_ava_srcs}
)

target_link_libraries(svgpu-manager
  ${manager_service_libraries}
  ${absl_flags_libraries}
  absl::symbolize
  absl::failure_signal_handler
  grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  nvidia-ml
)

set_target_properties(svgpu-manager
  PROPERTIES OUTPUT_NAME manager
)
add_compile_options(-Wall -Wextra -pedantic -Werror -Wno-write-strings)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/manager
  TYPE BIN
  RENAME svgpu_manager
)
