#include <absl/flags/parse.h>
#include <absl/flags/flag.h>

#include "SVGPUManager.hpp"

// arguments to this manager
ABSL_FLAG(uint32_t, manager_port, 3333, "(OPTIONAL) Specify manager port number");
ABSL_FLAG(uint32_t, worker_port_base, 4000, "(OPTIONAL) Specify base port number of API servers");
ABSL_FLAG(std::vector<std::string>, worker_argv, {}, "(OPTIONAL) Specify process arguments passed to API servers");
ABSL_FLAG(std::string, worker_path, "", "(REQUIRED) Specify API server binary path");
ABSL_FLAG(std::vector<std::string>, worker_env, {},
          "(OPTIONAL) Specify environment variables, e.g. HOME=/home/ubuntu, passed to API servers");
ABSL_FLAG(uint16_t, ngpus, 1, "(OPTIONAL) Number of GPUs the manager should use");
ABSL_FLAG(uint16_t, gpuoffset, 0, "(OPTIONAL)GPU id offset");
ABSL_FLAG(std::string, resmngr_addr, "", "(OPTIONAL) Address of the Alouatta resource manager. If enabled will run on grpc mode.");

int main(int argc, const char *argv[]) {
  absl::ParseCommandLine(argc, const_cast<char **>(argv));
  ava_manager::setupSignalHandlers();
  auto worker_argv = absl::GetFlag(FLAGS_worker_argv);
  auto worker_env = absl::GetFlag(FLAGS_worker_env);
  SVGPUManager* manager = new SVGPUManager(absl::GetFlag(FLAGS_manager_port), 
                      absl::GetFlag(FLAGS_worker_port_base),
                      absl::GetFlag(FLAGS_worker_path), 
                      worker_argv, worker_env,
                      absl::GetFlag(FLAGS_ngpus),
                      absl::GetFlag(FLAGS_gpuoffset));

  std::string rm_addr = absl::GetFlag(FLAGS_resmngr_addr);

  //normal ava mode
  if (rm_addr == "") {
    manager->RunServer();
  }
  //gRPC mode
  else {
    manager->RegisterSelf(rm_addr);
    manager->LaunchService();
  }
  
  return 0;
}
