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

int main(int argc, const char *argv[]) {
  absl::ParseCommandLine(argc, const_cast<char **>(argv));

  ava_manager::setupSignalHandlers();
  auto worker_argv = absl::GetFlag(FLAGS_worker_argv);
  auto worker_env = absl::GetFlag(FLAGS_worker_env);
  SVGPUManager manager(absl::GetFlag(FLAGS_manager_port), absl::GetFlag(FLAGS_worker_port_base),
                      absl::GetFlag(FLAGS_worker_path), worker_argv, worker_env);
  manager.RunServer();
  return 0;
}
