#include "SVGPUManager.hpp"
#include <boost/algorithm/string/join.hpp>
#include <sys/wait.h>

ava_proto::WorkerAssignReply SVGPUManager::HandleRequest(const ava_proto::WorkerAssignRequest &request) {
  ava_proto::WorkerAssignReply reply;

  // Start from input environment variables
  std::vector<std::string> environments(worker_env_);

  // Let first N GPUs visible
  if (request.gpu_count() > 0) {
    std::string visible_devices = "CUDA_VISIBLE_DEVICES=";
    for (uint32_t i = 0; i < request.gpu_count() ; ++i) {
      uint16_t cgpu = scheduler->getGPU();
      visible_devices += std::to_string(cgpu);
      if (i != request.gpu_count()-1) {
        visible_devices += ",";
      }
    }
    
    environments.push_back(visible_devices);
  }
  // Let API server use TCP channel
  environments.push_back("AVA_CHANNEL=TCP");

  // Pass port to API server
  auto port = worker_port_base_ + worker_id_.fetch_add(1, std::memory_order_relaxed);
  std::vector<std::string> parameters;
  parameters.push_back(std::to_string(port));

  // Append custom API server arguments
  for (const auto &argv : worker_argv_) {
    parameters.push_back(argv);
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

  reply.worker_address().push_back("0.0.0.0:" + std::to_string(port));

  return reply;
}
