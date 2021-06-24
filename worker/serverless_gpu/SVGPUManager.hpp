#ifndef __SVGPU_MANAGER_HPP__
#define __SVGPU_MANAGER_HPP__

#include "manager_service.hpp"
#include "manager_service.proto.h"
#include "scheduling/BaseScheduler.hpp"
#include "scheduling/RoundRobin.hpp"
#include <grpcpp/grpcpp.h>
#include "gpuserver.grpc.pb.h"
#include "resmngr.grpc.pb.h"

using ava_manager::ManagerServiceServerBase;
using grpc::Status;
using grpc::Server;
using grpc::Channel;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ClientContext;
using gpuserver::SpawnGPUWorkerRequest;
using gpuserver::SpawnGPUWorkerReply;
using gpuserver::GPU;
using resmngr::ResMngrService;
using resmngr::RegisterGPUNodeRequest;
using resmngr::RegisterGPUNodeResponse;

class SVGPUManager : public ManagerServiceServerBase, public GPU::Service {
  public:
  //fields
  BaseScheduler* scheduler;
  ResMngrClient* resmngr_client;

  //methods
  SVGPUManager(uint32_t port, uint32_t worker_port_base, std::string worker_path, std::vector<std::string> &worker_argv,
            std::vector<std::string> &worker_env, uint16_t ngpus, uint16_t gpu_offset)
    : ManagerServiceServerBase(port, worker_port_base, worker_path, worker_argv, worker_env) {
      // TODO: choose scheduler somehow
      scheduler = new RoundRobin(ngpus, gpu_offset);
    };

  void RegisterSelf(std::string rm_addr);
  void LaunchService();
  Status SpawnGPUWorker(ServerContext* context, const SpawnGPUWorkerRequest* request, SpawnGPUWorkerReply* response) override;
  ava_proto::WorkerAssignReply HandleRequest(const ava_proto::WorkerAssignRequest &request) override;

  class ResMngrClient {
    public:
      ResMngrClient(std::shared_ptr<Channel> channel)
      : stub_(ResMngrService::NewStub(channel)) {}
      
      //TODO: get gpus as arguments and set
      void RegisterSelf();

    private:
      std::unique_ptr<ResMngrService::Stub> stub_;
  };
};


#endif