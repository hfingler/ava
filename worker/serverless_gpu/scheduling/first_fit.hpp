#ifndef __ROUNDROBIN_SCHEDULER_HPP__
#define __ROUNDROBIN_SCHEDULER_HPP__

#include "common.hpp"

struct FirstFit : public BaseScheduler {

  FirstFit(
      std::map<uint32_t, std::map<uint32_t, GPUWorkerState>> *workers,
      std::map<uint32_t, GPUState> *gpu_states) :
    BaseScheduler(workers, gpu_states)
  {}

  int32_t getGPU() {
    for (auto& gpu_wks : *workers) {
        std::cerr << "checking gpu " << gpu_wks.first << std::endl;
        for (auto& port_wk : gpu_wks.second) {
            std::cerr << "checking port " << port_wk.first << std::endl;
            if (port_wk.second.busy == false) {
                port_wk.second.busy = true;
                return port_wk.first;
            }
        }
    }
    return -1;
  }
};

#endif