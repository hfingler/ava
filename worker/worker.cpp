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

#include "common/cmd_channel_impl.hpp"
#include "common/cmd_handler.hpp"
#include "common/socket.hpp"
#include "common/linkage.h"
#include "plog/Initializers/RollingFileInitializer.h"
#include "provision_gpu.h"
#include "worker_context.h"

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

int main(int argc, char *argv[]) {
  if (!(argc == 3 && !strcmp(argv[1], "migrate")) && (argc != 2)) {
    printf(
        "Usage: %s <listen_port>\n"
        "or     %s <mode> <listen_port> \n",
        argv[0], argv[0]);
    return 0;
  }
  absl::InitializeSymbolizer(argv[0]);

  /* Read GPU provision information. */
  char const *cuda_uuid_str = getenv("CUDA_VISIBLE_DEVICES");
  std::string cuda_uuid = cuda_uuid_str ? std::string(cuda_uuid_str) : "";
  char const *gpu_uuid_str = getenv("AVA_GPU_UUID");
  std::string gpu_uuid = gpu_uuid_str ? std::string(gpu_uuid_str) : "";
  char const *gpu_mem_str = getenv("AVA_GPU_MEMORY");
  std::string gpu_mem = gpu_mem_str ? std::string(gpu_mem_str) : "";
  provision_gpu = new ProvisionGpu(cuda_uuid, gpu_uuid, gpu_mem);

  /* setup signal handler */
  if ((original_sigint_handler = signal(SIGINT, sigint_handler)) == SIG_ERR) printf("failed to catch SIGINT\n");

  if ((original_sigsegv_handler = signal(SIGSEGV, sigsegv_handler)) == SIG_ERR) printf("failed to catch SIGSEGV\n");

  absl::FailureSignalHandlerOptions options;
  options.call_previous_handler = true;
  absl::InstallFailureSignalHandler(options);

  /* define arguments */
  auto wctx = ava::WorkerContext::instance();
  nw_worker_id = 0;
  unsigned int listen_port;

  /* parse arguments */
  listen_port = (unsigned int)atoi(argv[1]);
  wctx->set_api_server_listen_port(listen_port);
  std::cerr << "[worker#" << listen_port << "] To check the state of AvA remoting progress, use `tail -f " << wctx->log_file
            << "`" << std::endl;

  if (!getenv("AVA_CHANNEL") || !strcmp(getenv("AVA_CHANNEL"), "TCP")) {
    chan_hv = NULL;
    //create tcp socket
    chan = command_channel_socket_tcp_worker_new(listen_port);
    //create log file
    nw_record_command_channel = command_channel_log_new(listen_port);

    //this sets API id and other stuff
    init_internal_command_handler();

    char *rm_addr = std::getenv("RESMNGR_ADDR");
    //only loop if we are in serverless mode
    do {
      //get a guestlib connection
      std::cerr << "[worker#" << listen_port << "] waiting for connection" << std::endl;
      chan = command_channel_listen(chan);
      //this launches the thread that listens for commands
      init_command_handler(channel_create);
      std::cerr << "[worker#" << listen_port << "] got one, setting up cmd handler" << std::endl;
      wait_for_command_handler();
      destroy_command_handler(false);
      std::cerr << "[worker#" << listen_port << "] worker is done, looping." << std::endl;
    } while(rm_addr);

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
