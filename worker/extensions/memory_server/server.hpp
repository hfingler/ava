#ifndef __MEMORYSERVER_GPUSERVER_HPP__
#define __MEMORYSERVER_GPUSERVER_HPP__

#include <stdint.h>
#include <string>
#include <map>
#include <memory>
#include <thread>
#include <cuda_runtime_api.h>
#include "common/extensions/memory_server/common.hpp"

namespace GPUMemoryServer {
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
        std::map<std::string, std::map<uint64_t, std::unique_ptr<Allocation>>> allocs;
        //both these maps below are in MB unit
        std::map<std::string, uint64_t> used_memory;
        std::map<std::string, uint64_t> requested_memory;
        std::thread self_thread;

        Server(uint16_t gpu, std::string unix_socket_path) : gpu(gpu), unix_socket_path(unix_socket_path){}
        void handleRequest(char* buffer, void *responder);
        uint32_t handleMalloc(Request& req, Reply& rep);
        uint32_t handleFree(Request& req, Reply& rep);
        uint32_t handleFinish(Request& req, Reply& rep);
        uint32_t handleKernelIn(Request& req, Reply& rep);
        uint32_t handleKernelOut(Request& req, Reply& rep);
        uint32_t handleGetAllPointers(Request& req, Reply& rep);
        uint32_t handleMigrate(Request& req, Reply& rep);
        void run();
        void start() {
            self_thread = std::thread(&Server::run, this);
        }
    };

}

#endif
