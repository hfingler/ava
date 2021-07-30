#include "SVGPUManager.hpp"
#include <nvml.h>
#include <sys/wait.h>
#include <boost/algorithm/string/join.hpp>
#include "declaration.h"
#include "extensions/memory_server/server.hpp"

#include <string>
#include <memory>

/*************************************
 *
 *    gRPC methods
 *
 *************************************/

Status SVGPUManager::SpawnGPUWorker(ServerContext *AVA_UNUSED(context), const SpawnGPUWorkerRequest *request,
                                    SpawnGPUWorkerReply *response) {
  for (auto &pair : request->workers()) {
    uint32_t gpuid = pair.first;
    uint32_t n = pair.second;

    // TODO: make sure gpuid is sane

    std::cerr << "~~~ SpawnGPUWorker spawning " << n << " workers on GPU " << gpuid << std::endl;

    for (uint32_t i = 0; i < n; i++) {
      uint32_t port = LaunchWorker(gpuid);
      auto w = response->add_workers();
      w->set_id(gpuid);
      w->set_port(port);
    }
  }

  return Status::OK;
}

void SVGPUManager::RegisterSelf() {
  resmngr_client = new ResMngrClient(grpc::CreateChannel(resmngr_address, grpc::InsecureChannelCredentials()));
  resmngr_client->RegisterSelf(gpu_offset, n_gpus, gpu_memory);
}

void SVGPUManager::LaunchService() {
  std::string server_address("[::]:");
  if (const char *port = std::getenv("AVAMNGR_PORT")) {
    server_address += port;
  } else {
    std::cerr << "AVAMNGR_PORT not defined (you probably didnt run using task)" << std::endl;
    std::exit(1);
  }

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 1);
  builder.RegisterService(this);

  grpc_server = builder.BuildAndStart();
  std::cout << "Server listening on " << server_address << std::endl;
}

void SVGPUManager::setRealGPUOffsetCount() {
  nvmlReturn_t result;
  uint32_t device_count;
  result = nvmlInit();
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
    std::exit(1);
  }

  result = nvmlDeviceGetCount(&device_count);
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to query device count NVML: " << nvmlErrorString(result) << std::endl;
    std::exit(1);
  }

  result = nvmlShutdown();
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to shutdown NVML: " << nvmlErrorString(result) << std::endl;
    std::exit(1);
  }

  // bounds check requested gpus, use all gpus if n_gpus == 0
  if (n_gpus == 0) {
    device_count = device_count - gpu_offset;
  } 
  else {
    device_count = gpu_offset + n_gpus;
  }
}

void SVGPUManager::ResMngrClient::RegisterSelf(uint32_t& gpu_offset, uint32_t& n_gpus, std::map<uint32_t, uint64_t> &gpu_memory) {
  RegisterGPUNodeRequest request;
  nvmlReturn_t result;

  result = nvmlInit();
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
    std::exit(1);
  }

  // register each gpu with its available memory
  for (unsigned int i = gpu_offset; i < gpu_offset+n_gpus; i++) {
    nvmlDevice_t device;
    nvmlMemory_t memory;

    result = nvmlDeviceGetHandleByIndex(i, &device);
    if (result != NVML_SUCCESS) {
      std::cerr << "Failed to get handle for device " << i << ": " << nvmlErrorString(result) << std::endl;
      std::exit(1);
    }

    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (result != NVML_SUCCESS) {
      std::cerr << "Failed to get memory of device " << i << ": " << nvmlErrorString(result) << std::endl;
      std::exit(1);
    }

    //store locally
    gpu_memory[i] = memory.free;

    RegisterGPUNodeRequest::GPU *g = request.add_gpus();
    g->set_id(i);
    g->set_memory(memory.free);
  }

  result = nvmlShutdown();
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to shutdown NVML: " << nvmlErrorString(result) << std::endl;
    std::exit(1);
  }

  ClientContext context;
  RegisterGPUNodeResponse reply;
  Status status = stub_->RegisterGPUNode(&context, request, &reply);

  if (!status.ok()) {
    std::cerr << "Error registering self with resmngr:" << status.error_code() << ": " << status.error_message()
              << std::endl;
    std::exit(1);
  }
}

/*************************************
 *
 *    AvA HandleRequest override
 *
 *************************************/

ava_proto::WorkerAssignReply SVGPUManager::HandleRequest(const ava_proto::WorkerAssignRequest &request) {
  ava_proto::WorkerAssignReply reply;

  if (request.gpu_count() > 1) {
    std::cerr << "ERR: someone requested more than 1 GPU, no bueno" << std::endl;
    return reply;
  }

  uint16_t cgpu = scheduler->getGPU();
  uint32_t port = LaunchWorker(cgpu);
  reply.worker_address().push_back("0.0.0.0:" + std::to_string(port));

  return reply;
}

/*************************************
 *
 *    General stuff
 *
 *************************************/

SVGPUManager::SVGPUManager(uint32_t port, uint32_t worker_port_base, std::string worker_path, std::vector<std::string> &worker_argv,
            std::vector<std::string> &worker_env, uint16_t ngpus, uint16_t gpu_offset)
    : ManagerServiceServerBase(port, worker_port_base, worker_path, worker_argv, worker_env) {
  this->scheduler = new RoundRobin(ngpus, gpu_offset);
  this->n_gpus = ngpus;
  this->gpu_offset = gpu_offset;
  this->uuid_counter = 0;
  //already update to real values using nvml
  setRealGPUOffsetCount();
};

void SVGPUManager::LaunchMemoryServers() {
  std::string base_path = GPUMemoryServer::get_base_socket_path();

  for (unsigned int i = gpu_offset; i < gpu_offset+n_gpus; i++) {
    std::ostringstream stringStream;
    stringStream << base_path << i;
    std::cerr << "Launching GPU memory server for GPU " << i << " at socket "
          << stringStream.str() << std::endl;

    auto sv = std::make_unique<GPUMemoryServer::Server>
        (i, gpu_memory[i], stringStream.str(), resmngr_address);
    sv->start();
    memory_servers.push_back(std::move(sv));
  }
}

uint32_t SVGPUManager::LaunchWorker(uint32_t gpu_id) {
  // Start from input environment variables
  std::vector<std::string> environments(worker_env_);

  for (std::string &e : environments) {
    printf("   %s", e.c_str());
  }

  std::string visible_devices = "GPU_DEVICE=" + std::to_string(gpu_id);
  environments.push_back(visible_devices);

  // Let API server use TCP channel
  environments.push_back("AVA_CHANNEL=TCP");

  std::string worker_uuid = "AVA_WORKER_UUID=" + std::to_string(uuid_counter);
  environments.push_back(worker_uuid);
  uuid_counter++;

  // Pass port to API server
  auto port = worker_port_base_ + worker_id_.fetch_add(1, std::memory_order_relaxed);
  std::vector<std::string> parameters;
  parameters.push_back(std::to_string(port));

  // Append custom API server arguments
  for (const auto &argv : worker_argv_) {
    parameters.push_back(argv);
  }

  for (auto& element : environments) {
    printf("  > %s\n", element.c_str());
  }

  std::cerr << "Spawn API server at 0.0.0.0:" << port << " (cmdline=\"" << boost::algorithm::join(environments, " ")
            << " " << boost::algorithm::join(parameters, " ") << "\")" << std::endl;

  auto child_pid = SpawnWorker(environments, parameters);

  auto child_monitor = std::make_shared<std::thread>(
      [](pid_t child_pid, uint32_t port, std::map<pid_t, std::shared_ptr<std::thread>> *worker_monitor_map) {
        pid_t ret = waitpid(child_pid, NULL, 0);
        std::cerr << "[pid=" << child_pid << "] API server at ::" << port << " has exit (waitpid=" << ret << ")"
                  << std::endl;
        worker_monitor_map->erase(port);
      },
      child_pid, port, &worker_monitor_map_);
  child_monitor->detach();
  worker_monitor_map_.insert({port, child_monitor});

  return port;
}
