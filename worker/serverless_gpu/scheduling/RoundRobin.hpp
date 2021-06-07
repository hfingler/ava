#ifndef __ROUNDROBIN_SCHEDULER_HPP__
#define __ROUNDROBIN_SCHEDULER_HPP__

#include "BaseScheduler.hpp"

class RoundRobin : public BaseScheduler {
  public:
  uint16_t current_gpu;

  RoundRobin(uint16_t ngpus) 
    : BaseScheduler(ngpus), current_gpu(0) {};

  int getGPU() {
    int gpu = current_gpu % ngpus;
    printf("***************Returning gpu %d\n", gpu);
    current_gpu++;
    return gpu;
  }
};

#endif