#ifndef __SVGPU_MANAGER_HPP__
#define __SVGPU_MANAGER_HPP__

#include <grpcpp/grpcpp.h>

#include <map>
#include <vector>

#include "manager_service.hpp"
#include "manager_service.proto.h"
#include "resmngr.grpc.pb.h"
#include "resmngr.pb.h"
#include "extensions/memory_server/common.hpp"

#include "scheduling/common.hpp"

using ava_manager::ManagerServiceServerBase;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using resmngr::ResMngrService;
using namespace GPUMemoryServer;

struct SVGPUManager : public ManagerServiceServerBase {
  /**************************************
   *        INTERNAL CLASSES
   **************************************/
    struct ResMngrClient {
        ResMngrClient(std::shared_ptr<Channel> channel) : stub_(ResMngrService::NewStub(channel)) {}
        std::string registerSelf();
        void addGPUWorker(std::string uuid);
        std::unique_ptr<ResMngrService::Stub> stub_;
    };


    /**************************************
     *        FIELDS
     **************************************/
    // gRPC related fields
    std::string resmngr_address;
    ResMngrClient *resmngr_client;
    std::string uuid;

    // internal state
    uint32_t n_gpus, gpu_offset;
    uint32_t uuid_counter;

    // GPU and worker information
    BaseScheduler *scheduler;
    std::thread central_server_thread;
    void* zmq_context;
    void* zmq_central_socket;

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
    uint32_t launchWorker(uint32_t gpu_id);
    void createScheduler(std::string name);
    //ava override
    ava_proto::WorkerAssignReply HandleRequest(const ava_proto::WorkerAssignRequest &request) override;

    void handleRequest(Request& req, Reply& rep);
    void handleMalloc(Request& req, Reply& rep);
    void handleFree(Request& req, Reply& rep);
    void handleRequestedMemory(Request& req, Reply& rep);
    void handleFinish(Request& req, Reply& rep);
    void handleKernelIn(Request& req, Reply& rep);
    void handleKernelOut(Request& req, Reply& rep);
    void handleReady(Request& req, Reply& rep);
    void handleSchedule(Request& req, Reply& rep);
};

#endif
