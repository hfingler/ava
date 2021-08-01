#include <stdlib.h>
#include <absl/debugging/symbolize.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include <chrono>
#include <thread>

#include "SVGPUManager.hpp"

// arguments to this manager
ABSL_FLAG(uint32_t, manager_port, 3333, "(OPTIONAL) Specify manager port number");
ABSL_FLAG(uint32_t, worker_port_base, 4000, "(OPTIONAL) Specify base port number of API servers");
ABSL_FLAG(std::vector<std::string>, worker_argv, {}, "(OPTIONAL) Specify process arguments passed to API servers");
ABSL_FLAG(std::string, worker_path, "", "(REQUIRED) Specify API server binary path");
ABSL_FLAG(std::vector<std::string>, worker_env, {},
          "(OPTIONAL) Specify environment variables, e.g. HOME=/home/ubuntu, passed to API servers");
ABSL_FLAG(uint16_t, ngpus, 0, "(OPTIONAL) Number of GPUs the manager should use");
ABSL_FLAG(uint16_t, gpuoffset, 0, "(OPTIONAL)GPU id offset");
ABSL_FLAG(std::string, resmngr_addr, "",
          "(OPTIONAL) Address of the Alouatta resource manager. If enabled will run on grpc mode.");

ABSL_FLAG(std::string, allctx, "no", "(OPTIONAL) turn on setting up all device ctx on workers (required for migration)");
ABSL_FLAG(std::string, reporting, "no", "(OPTIONAL) turn on client reports to gpu server (required for migration)");
ABSL_FLAG(std::string, debug_migration, "no", "(OPTIONAL) turn on debug migration");

int main(int argc, const char *argv[]) {
  absl::ParseCommandLine(argc, const_cast<char **>(argv));
  absl::InitializeSymbolizer(argv[0]);
  ava_manager::setupSignalHandlers();
  auto worker_argv = absl::GetFlag(FLAGS_worker_argv);
  auto worker_env = absl::GetFlag(FLAGS_worker_env);

  uint32_t port;
  // let's give env priority
  if (const char *env_port = std::getenv("AVAGPU_PORT")) {
    port = static_cast<uint32_t>(std::stoul(env_port));
  } else {
    port = absl::GetFlag(FLAGS_manager_port);
  }

  //check for debug flag
  if (absl::GetFlag(FLAGS_debug_migration) != "no") {
    printf(">>> Setting SG_DEBUG_MIGRATION \n");
    std::string kmd = "SG_DEBUG_MIGRATION=1";
    worker_env.push_back(kmd);
    setenv("SG_DEBUG_MIGRATION", "1", 1);
  }

  char *rm_addr = std::getenv("RESMNGR_ADDR");
  if (rm_addr) {
    //let's just rename this thing since a lot of parts will use it
    setenv("SERVERLESS_MODE", "1", 1);
    std::string kmd = "SERVERLESS_MODE=1";
    worker_env.push_back(kmd);
  }

  /*
   *  Check all contexts option
   */
  if (absl::GetFlag(FLAGS_allctx) == "yes") {
    std::cerr << "[SVLESS-MNGR]: All context init is enabled." << std::endl;
    worker_env.push_back("AVA_ENABLE_ALL_CTX=yes");
  }
  else {
    std::cerr << "[SVLESS-MNGR]: All context init is DISABLE." << std::endl;
    worker_env.push_back("AVA_ENABLE_ALL_CTX=no");
  }

  /*
   *  Check reporting option
   */
  if (absl::GetFlag(FLAGS_reporting) == "yes") {
    std::cerr << "[SVLESS-MNGR]: Reporting is enabled, launching GPU Servers.." << std::endl;
    worker_env.push_back("AVA_ENABLE_REPORTING=yes");
  }
  else {
    std::cerr << "[SVLESS-MNGR]: Reporting is not enabled, no GPU servers will be launched" << std::endl;
    worker_env.push_back("AVA_ENABLE_REPORTING=no");
  }

  /*
   *  Create the manager 
   */
  std::cerr << "[SVLESS-MNGR]: Using port " << port << " for AvA" << std::endl;
  SVGPUManager *manager =
      new SVGPUManager(port, absl::GetFlag(FLAGS_worker_port_base), absl::GetFlag(FLAGS_worker_path), worker_argv,
                       worker_env, absl::GetFlag(FLAGS_ngpus), absl::GetFlag(FLAGS_gpuoffset));

  if (rm_addr) {
    std::string full_addr(rm_addr);
    full_addr += ":";
    full_addr += std::getenv("RESMNGR_PORT");
    std::cerr << "[SVLESS-MNGR]: Running manager on serverless mode, rm at " << full_addr << std::endl;
    manager->resmngr_address = full_addr;
  }
  
  if (absl::GetFlag(FLAGS_reporting) == "yes") {
    manager->LaunchMemoryServers();
    std::cerr << "[SVLESS-MNGR]:Launched memory servers, sleeping to give them time to spin up" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }

  // normal ava mode
  if (!rm_addr) {
    std::cerr << "[SVLESS-MNGR]: Running manager on normal manager mode" << std::endl;
    manager->RunServer();
  }
  // gRPC mode
  else {
    manager->LaunchService();
    // wait a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    manager->RegisterSelf();

    //launch the memory servers, needs to be after LaunchService, which fills device count
    manager->LaunchMemoryServers();

    // block forever
    manager->grpc_server->Wait();
  }

  return 0;
}
