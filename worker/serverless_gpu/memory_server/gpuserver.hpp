#ifndef __MEMORYSERVER_GPUSERVER_HPP__
#define __MEMORYSERVER_GPUSERVER_HPP__

#include <stdint.h>
#include <string>
#include <map>
#include <memory>

#include <cuda_runtime_api.h>

namespace GPUMemoryServer {

    enum RequestType { ALLOC, FREE, FINISHED };

    union RequestData {
        uint64_t size;
        void* devPtr;
    };

    struct Request {
        RequestType type;
        uint32_t worker_id;
        RequestData data;
    };

    struct Reply {
        cudaIpcMemHandle_t memHandle;
    };

    struct Allocation {
        uint64_t size;
        void* devPtr;

        Allocation(uint32_t size, void* devPtr)
            : size(size), devPtr(devPtr) {}

        ~Allocation();
    };

    class Server {
        public:
        //fields
        uint16_t gpu;
        std::string unix_socket_path;
        //map of worker_id to a map of dev_pointers to allocation struct
        std::map<uint32_t, std::map<uint64_t, std::unique_ptr<Allocation>>> allocs;
        std::map<uint32_t, uint64_t> used_memory;

        Server(uint16_t gpu, std::string unix_socket_path) : gpu(gpu), unix_socket_path(unix_socket_path){}
        void run();
    };


}

#endif