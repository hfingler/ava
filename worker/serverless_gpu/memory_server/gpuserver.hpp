#ifndef __MEMORYSERVER_GPUSERVER_HPP__
#define __MEMORYSERVER_GPUSERVER_HPP__

#include <stdint.h>
#include <string>

#include <cuda_runtime_api.h>

namespace GPUMemoryServer {

    enum RequestType { ALLOC, FREE };

    struct Request {
        RequestType type;
        uint32_t size;
    };

    struct Reply {
        cudaIpcMemHandle_t memHandle;
    };

    struct Allocation {
        uint32_t size;
    };


    class Server {
        public:
        uint16_t gpu;
        std::string unix_socket_path;
        Server(uint16_t gpu, std::string unix_socket_path) : gpu(gpu), unix_socket_path(unix_socket_path){}
        void run();
    };


}

#endif