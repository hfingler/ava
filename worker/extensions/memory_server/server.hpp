#ifndef __MEMORYSERVER_GPUSERVER_HPP__
#define __MEMORYSERVER_GPUSERVER_HPP__

#include <stdint.h>
#include <string>
#include <map>
#include <memory>
#include <thread>
#include "extensions/memory_server/common.hpp"

namespace GPUMemoryServer {
    struct WorkerInfo {
        uint64_t mem_used;
        uint64_t requested_memory;
        WorkerInfo()
            : mem_used(0) {}
    };

    struct Server {
        //fields
        std::string self_unix_socket_path;
        std::string central_server_unix_socket_path;
        void* central_socket;

        uint16_t gpu;
        //uint64_t gpu_memory_total;
        uint64_t gpu_memory_used;
        uint32_t kernels_queued;
        std::map<std::string, WorkerInfo> workers_info;
        
        std::thread self_thread;

        Server(uint16_t gpu, std::string self_unix_socket_path, std::string central_server_unix_socket_path);
        void handleRequest(Request& req, Reply& rep);
        void handleMalloc(Request& req, Reply& rep);
        void handleFree(Request& req, Reply& rep);
        void handleRequestedMemory(Request& req, Reply& rep);
        void handleFinish(Request& req, Reply& rep);
        void handleKernelIn(Request& req, Reply& rep);
        void handleKernelOut(Request& req, Reply& rep);
        void handleReady(Request& req, Reply& rep);

        void run();
        void start() {
            self_thread = std::thread(&Server::run, this);
        }
    };
}

#endif
