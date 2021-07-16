#ifndef __SVGPU_MANAGER_HPP__
#define __SVGPU_MANAGER_HPP__

#include <grpcpp/grpcpp.h>
#include <vector>
#include "gpuserver.grpc.pb.h"
#include "manager_service.hpp"
#include "manager_service.proto.h"
#include "resmngr.grpc.pb.h"
#include "scheduling/BaseScheduler.hpp"
#include "scheduling/RoundRobin.hpp"
#include "extensions/memory_server/server.hpp"

using ava_manager::ManagerServiceServerBase;
using gpuserver::GPU;
using gpuserver::SpawnGPUWorkerReply;
using gpuserver::SpawnGPUWorkerRequest;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using resmngr::RegisterGPUNodeRequest;
using resmngr::RegisterGPUNodeResponse;
using resmngr::ResMngrService;

class SVGPUManager : public ManagerServiceServerBase, public GPU::Service {
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

  // fields
  BaseScheduler *scheduler;
  ResMngrClient *resmngr_client;
  std::unique_ptr<Server> grpc_server;
  uint32_t n_gpus;
  uint32_t gpu_offset;
  bool serverless_mode;
  std::vector<std::unique_ptr<GPUMemoryServer::Server>> memory_servers;
  std::string memory_mode;

  // methods
  SVGPUManager(uint32_t port, uint32_t worker_port_base, std::string worker_path, std::vector<std::string> &worker_argv,
            std::vector<std::string> &worker_env, uint16_t ngpus, uint16_t gpu_offset, std::string memory_mode)
    : ManagerServiceServerBase(port, worker_port_base, worker_path, worker_argv, worker_env) {
      // TODO: choose scheduler somehow
      scheduler = new RoundRobin(ngpus, gpu_offset);
      n_gpus = ngpus;
      gpu_offset = gpu_offset;
      memory_mode = memory_mode;
    };

  void RegisterSelf(std::string rm_addr);
  void LaunchService();
  void LaunchMemoryServers();
  Status SpawnGPUWorker(ServerContext *context, const SpawnGPUWorkerRequest *request,
                        SpawnGPUWorkerReply *response) override;
  ava_proto::WorkerAssignReply HandleRequest(const ava_proto::WorkerAssignRequest &request) override;
  uint32_t LaunchWorker(uint32_t gpu_id);
};

#endif
