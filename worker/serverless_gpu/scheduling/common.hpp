#ifndef __SCHEDULING_COMMON_HPP__
#define __SCHEDULING_COMMON_HPP__

#include <stdint.h>

struct GPUWorkerState {
        uint32_t port;
        bool busy;
};

struct GPUState {
    std::vector<GPUWorkerState> workers;
    uint64_t total_memory;
    uint64_t used_memory;
};

struct BaseScheduler {
  std::map<uint32_t, std::map<uint32_t, GPUWorkerState>> *workers;
  std::map<uint32_t, GPUState> *gpu_states;

  BaseScheduler(
      std::map<uint32_t, std::map<uint32_t, GPUWorkerState>> *workers,
      std::map<uint32_t, GPUState> *gpu_states) :
    workers(workers), gpu_states(gpu_states)
  {}

  virtual int32_t getGPU() = 0;
};


#endif