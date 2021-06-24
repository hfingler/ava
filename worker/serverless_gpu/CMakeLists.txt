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
get_filename_component(hw_proto "pb/gpuserver.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/gpuserver.pb.cc")
set(proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/gpuserver.pb.h")
set(grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/gpuserver.grpc.pb.cc")
set(grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/gpuserver.grpc.pb.h")

add_custom_command(
  OUTPUT "${proto_srcs}" "${proto_hdrs}" "${grpc_srcs}" "${grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS --grpc_out "."
    --cpp_out "."
    -I "${hw_proto_path}"
    --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
    "${hw_proto}"
  DEPENDS "${hw_proto}")

add_library(grpc_proto
  ${grpc_srcs}
  ${grpc_hdrs}
  ${proto_srcs}
  ${proto_hdrs})

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
  grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
)
set_target_properties(svgpu-manager
  PROPERTIES OUTPUT_NAME manager
)
add_compile_options(-Wall -Wextra -pedantic -Werror -Wno-write-strings)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/manager
  TYPE BIN
  RENAME svgpu_manager
)
