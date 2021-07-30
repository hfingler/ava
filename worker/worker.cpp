#include "worker.h"

#include <absl/debugging/failure_signal_handler.h>
#include <absl/debugging/symbolize.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <fmt/core.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <gsl/gsl>
#include <iostream>
#include <string>

#include <zmq.h>
#include <nvml.h>

#include "common/cmd_channel_impl.hpp"
#include "common/cmd_handler.hpp"
#include "common/socket.hpp"
#include "common/linkage.h"
#include "plog/Initializers/RollingFileInitializer.h"
#include "worker_context.h"
#include "common/common_context.h"

#include "extensions/memory_server/client.hpp"


struct command_channel *chan;
struct command_channel *chan_hv = NULL;
extern int nw_global_vm_id;

__sighandler_t original_sigint_handler = SIG_DFL;
__sighandler_t original_sigsegv_handler = SIG_DFL;
__sighandler_t original_sigchld_handler = SIG_DFL;

void sigint_handler(int signo) {
  void *array[10];
  size_t size;
  size = backtrace(array, 10);
  fprintf(stderr, "===== backtrace =====\n");
  fprintf(stderr, "receive signal %d:\n", signo);
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  if (chan) {
    command_channel_free(chan);
    chan = NULL;
  }
  signal(signo, original_sigint_handler);
  raise(signo);
}

void sigsegv_handler(int signo) {
  void *array[10];
  size_t size;
  size = backtrace(array, 10);
  fprintf(stderr, "===== backtrace =====\n");
  fprintf(stderr, "receive signal %d:\n", signo);
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  if (chan) {
    command_channel_free(chan);
    chan = NULL;
  }
  signal(signo, original_sigsegv_handler);
  raise(signo);
}

void nw_report_storage_resource_allocation(const char *const name, ssize_t amount) {
  if (chan_hv) command_channel_hv_report_storage_resource_allocation(chan_hv, name, amount);
}

void nw_report_throughput_resource_consumption(const char *const name, ssize_t amount) {
  if (chan_hv) command_channel_hv_report_throughput_resource_consumption(chan_hv, name, amount);
}

static struct command_channel *channel_create() { return chan; }

EXPORTED_WEAKLY std::string worker_init_log() {
  std::ios_base::Init();
  // Initialize logger
  std::string log_file = std::tmpnam(nullptr);
  plog::init(plog::debug, log_file.c_str());
  std::cerr << "To check the state of AvA remoting progress, use `tail -f " << log_file << "`" << std::endl;
  return log_file;
}

static void create_cuda_contexts() {
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

  __internal_setDeviceCount(device_count);

  for (int i = 0 ; i < device_count ; i++) {
    cudaSetDevice(i);
    //this forcibly creates a primary context, which is lazily-created
    cudaFree(0);
    std::cerr << "Created CUDA context on device [" << i << "]" << std::endl;
  }
}

int main(int argc, char *argv[]) {
  if (!(argc == 3 && !strcmp(argv[1], "migrate")) && (argc != 2)) {
    printf(
        "Usage: %s <listen_port>\n"
        "or     %s <mode> <listen_port> \n",
        argv[0], argv[0]);
    return 0;
  }
  absl::InitializeSymbolizer(argv[0]);

  char const *gpu_device_str = getenv("GPU_DEVICE");
  std::string gpu_device = std::string(gpu_device_str);

  //AVA_WORKER_UUID is a unique, starting at 0, id we can use
  char const *cworker_uuid = getenv("AVA_WORKER_UUID");
  std::string worker_uuid = std::string(cworker_uuid);

  char *cmmode = std::getenv("GPU_MEMORY_MODE");
  std::string mmode = cmmode ? std::string(cmmode) : "default";
  
  // preemptively create context on all GPUs
  create_cuda_contexts();

  /* set current device*/
  GPUMemoryServer::Client::getInstance().setCurrentGPU(std::stoi(gpu_device));
  auto ccontext = ava::CommonContext::instance();
  ccontext->current_device = std::stoi(gpu_device);

  /* setup signal handler */
  if ((original_sigint_handler = signal(SIGINT, sigint_handler)) == SIG_ERR) printf("failed to catch SIGINT\n");
  if ((original_sigsegv_handler = signal(SIGSEGV, sigsegv_handler)) == SIG_ERR) printf("failed to catch SIGSEGV\n");

  absl::FailureSignalHandlerOptions options;
  options.call_previous_handler = true;
  absl::InstallFailureSignalHandler(options);

  /* define arguments */
  auto wctx = ava::WorkerContext::instance();
  unsigned int listen_port;
  nw_worker_id = 0;

  /* parse arguments */
  listen_port = (unsigned int)atoi(argv[1]);
  wctx->set_api_server_listen_port(listen_port);
  std::cerr << "[worker#" << listen_port << "] To check the state of AvA remoting progress, use `tail -f " << wctx->log_file
            << "`" << std::endl;

  GPUMemoryServer::Client::getInstance().connectToGPUs();

  if (!getenv("AVA_CHANNEL") || !strcmp(getenv("AVA_CHANNEL"), "TCP")) {
    chan_hv = NULL;
    chan = command_channel_socket_tcp_worker_new(listen_port);
    nw_record_command_channel = command_channel_log_new(listen_port);
    //this sets API id and other stuff
    init_internal_command_handler();

    //only loop if we are in serverless mode
    do {
      //get a guestlib connection
      std::cerr << "[worker#" << listen_port << "] waiting for connection" << std::endl;
      chan = command_channel_listen(chan);
      //this launches the thread that listens for commands
      init_command_handler(channel_create);

      //TODO: I can see a race condition with init_command_handler and commands below, need to be sure and fix

      //if this is serverless, we need to update our id
      if (svless_vmid == "NO_VMID" || svless_vmid == "") {
        printf("svless_vmid is default, using %s\n", worker_uuid.c_str());
        GPUMemoryServer::Client::getInstance().setUuid(worker_uuid);
      }
      else {
        printf("got vmid from cmd channel: %s\n", svless_vmid.c_str());
        GPUMemoryServer::Client::getInstance().setUuid(svless_vmid);
      }

      //report our max memory requested
      GPUMemoryServer::Client::getInstance().sendMemoryRequestedValue(requested_gpu_mem);
      //GPUMemoryServer::Client::getInstance().sendMemoryRequestedValue(16);

      std::cerr << "[worker#" << listen_port << "] got one, setting up cmd handler" << std::endl;
      wait_for_command_handler();
      destroy_command_handler(false);
      std::cerr << "[worker#" << listen_port << "] worker is done, looping." << std::endl;

      //clean up allocations, local and remote
      GPUMemoryServer::Client::getInstance().cleanup();
      //go back to original GPU
      GPUMemoryServer::Client::getInstance().setOriginalGPU();
    } while(std::getenv("SERVERLESS_MODE"));

    std::cerr << "[worker#" << listen_port << "] freeing channel and quiting." << std::endl;
    command_channel_free(chan);
    command_channel_free((struct command_channel *)nw_record_command_channel);
    return 0;
  } else {
    printf("Unsupported AVA_CHANNEL type (export AVA_CHANNEL=[TCP]\n");
    return 0;
  }

  //nw_record_command_channel = command_channel_log_new(listen_port);
  init_internal_command_handler();
  init_command_handler(channel_create);
  LOG_INFO << "[worker#" << listen_port << "] start polling tasks";
  wait_for_command_handler();
  command_channel_free(chan);
  command_channel_free((struct command_channel *)nw_record_command_channel);
  if (chan_hv) command_channel_hv_free(chan_hv);

  return 0;
}
