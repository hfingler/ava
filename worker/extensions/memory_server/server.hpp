#ifndef __MEMORYSERVER_GPUSERVER_HPP__
#define __MEMORYSERVER_GPUSERVER_HPP__

#include <stdint.h>
#include <string>
#include <map>
#include <memory>
#include <thread>
#include <grpcpp/grpcpp.h>
#include "resmngr.grpc.pb.h"
#include "extensions/memory_server/common.hpp"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using resmngr::RegisterGPUNodeRequest;
using resmngr::RegisterGPUNodeResponse;
using resmngr::ResMngrService;

namespace GPUMemoryServer {
    struct WorkerInfo {
        uint64_t mem_used;
        uint64_t requested_memory;
        WorkerInfo()
            : mem_used(0) {}
        bool isGoodCitizen() {
            return mem_used <= requested_memory;
        }
    };

    class Server {
        public:
        class ResMngrClient {
            public:
            ResMngrClient(std::shared_ptr<Channel> channel)
            : stub_(ResMngrService::NewStub(channel)) {}
            
            //TODO: get gpus as arguments and set
            void RegisterSelf(uint32_t& gpu_offset, uint32_t& n_gpus);

            private:
            std::unique_ptr<ResMngrService::Stub> stub_;
        };
        ResMngrClient *resmngr_client;

        //fields
        std::string unix_socket_path;
        uint16_t gpu;
        uint64_t gpu_memory_total;
        uint64_t gpu_memory_used;
        uint32_t kernels_queued;
        std::map<std::string, WorkerInfo> workers_info;
        
        std::thread self_thread;

        Server(uint16_t gpu, uint64_t total_memory, std::string unix_socket_path, std::string resmngr_address);
        void handleRequest(Request& req, Reply& rep);
        void handleMalloc(Request& req, Reply& rep);
        void handleFree(Request& req, Reply& rep);
        void handleRequestedMemory(Request& req, Reply& rep);
        void handleFinish(Request& req, Reply& rep);
        void handleKernelIn(Request& req, Reply& rep);
        void handleKernelOut(Request& req, Reply& rep);
        void run();
        void start() {
            self_thread = std::thread(&Server::run, this);
        }
    };

}

#endif
