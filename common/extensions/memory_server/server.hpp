#ifndef __MEMORYSERVER_GPUSERVER_HPP__
#define __MEMORYSERVER_GPUSERVER_HPP__

#include <stdint.h>
#include <string>
#include <map>
#include <memory>
#include <thread>
#include <cuda_runtime_api.h>
#include "common.hpp"

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
        void handleRequest(char* buffer, Reply* rep);
        void handleMalloc(Request& req, Reply* rep);
        void handleFree(Request& req, Reply* rep);
        void handleFinish(Request& req, Reply* rep);
        void handleKernelIn(Request& req, Reply* rep);
        void handleKernelOut(Request& req, Reply* rep);
        void run();
        void start() {
            self_thread = std::thread(&Server::run, this);
        }
    };

}

#endif