#ifndef __SVGPU_MANAGER_HPP__
#define __SVGPU_MANAGER_HPP__

#include <grpcpp/grpcpp.h>

#include <map>
#include <vector>

#include "manager_service.hpp"
#include "manager_service.proto.h"
#include "resmngr.grpc.pb.h"
#include "scheduling/BaseScheduler.hpp"
#include "scheduling/RoundRobin.hpp"
#include "extensions/memory_server/server.hpp"
#include "extensions/memory_server/common.hpp"

using ava_manager::ManagerServiceServerBase;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using resmngr::RegisterGPUNodeRequest;
using resmngr::RegisterGPUNodeResponse;
using resmngr::ResMngrService;

struct SVGPUManager : public ManagerServiceServerBase {
  /**************************************
   *        INTERNAL CLASSES
   **************************************/
    struct ResMngrClient {
        ResMngrClient(std::shared_ptr<Channel> channel) : stub_(ResMngrService::NewStub(channel)) {}
        // TODO: get gpus as arguments and set
        void registerSelf();
        std::unique_ptr<ResMngrService::Stub> stub_;
    };

    struct GPUWorkerState {
        uint32_t port;
        bool busy;
    };

    struct GPUState {
        std::vector<GPUWorkerState> workers;
        uint64_t total_memory;
        uint64_t used_memory;
    };

    /**************************************
     *        FIELDS
     **************************************/
    // gRPC related fields
    std::string resmngr_address;
    ResMngrClient *resmngr_client;

    // internal state
    uint32_t n_gpus, gpu_offset;
    uint32_t uuid_counter;

    // GPU and worker information
    BaseScheduler *scheduler;
    std::vector<std::unique_ptr<GPUMemoryServer::Server>> memory_servers;
    std::thread central_server_thread;
    void* zmq_context;
    void* zmq_central_socket;
    std::string central_server_unix_socket_path;

    std::map<uint32_t, GPUState> gpu_states;
    std::map<uint32_t, std::map<uint32_t, GPUWorkerState>> gpu_workers;

    uint32_t precreated_workers;

    /**************************************
     *        METHODS
     **************************************/
    SVGPUManager(uint32_t port, uint32_t worker_port_base, std::string worker_path, 
            std::vector<std::string> &worker_argv, std::vector<std::string> &worker_env, 
            uint16_t ngpus, uint16_t gpu_offset, std::string resmngr_address, 
            std::string scheduler_name, uint32_t precreated_workers);

    void setRealGPUOffsetCount();
    void registerSelf();
    void launchReportServers();
    void centralManagerLoop();
    void handleRequest(GPUMemoryServer::Request& req, GPUMemoryServer::Reply& rep);
    uint32_t launchWorker(uint32_t gpu_id);
    //ava overrides
    ava_proto::WorkerAssignReply HandleRequest(const ava_proto::WorkerAssignRequest &request) override;
};

#endif
