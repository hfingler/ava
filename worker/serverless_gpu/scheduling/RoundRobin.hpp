#ifndef __ROUNDROBIN_SCHEDULER_HPP__
#define __ROUNDROBIN_SCHEDULER_HPP__

#include "BaseScheduler.hpp"

class RoundRobin : public BaseScheduler {
 public:
  uint16_t gpu_offset;
  uint16_t current_gpu;

  RoundRobin(uint16_t ngpus, uint16_t gpu_offset) : BaseScheduler(ngpus), gpu_offset(gpu_offset), current_gpu(0){};

  int getGPU() {
    int gpu = gpu_offset + (current_gpu % ngpus);
    printf("***************Returning gpu %d\n", gpu);
    current_gpu++;
    return gpu;
  }
};

#endif