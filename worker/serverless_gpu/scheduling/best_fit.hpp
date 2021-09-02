#ifndef __BESTFIT_SCHEDULER_HPP__
#define __BESTFIT_SCHEDULER_HPP__

#include "common.hpp"
#include <map>

struct BestFit : public BaseScheduler {

  BestFit(
      std::map<uint32_t, std::map<uint32_t, GPUWorkerState>> *workers,
      std::map<uint32_t, GPUState> *gpu_states) :
    BaseScheduler(workers, gpu_states)
  {}

  int32_t getGPU(uint64_t requested_memory) {
    int32_t best_gpu = -1;
    uint64_t best_mem = -1;

    for (auto& gpus : *gpu_states) {
      uint64_t free_memory = (gpus.second.total_memory - gpus.second.used_memory);
      std::cerr << "gpu " << gpus.first << " has free memory: " << free_memory <<std::endl;
      //if there is enough memory
      if (requested_memory <= free_memory) {
        //and if there is an idle worker on this gpu
        bool free_worker_exists = false;
        for (auto& gpu_wks : (*workers)[gpus.first]) {
          if (gpu_wks.second.busy == false) {
            free_worker_exists = true;
          }
        }
        
        if (free_worker_exists) {
          if (best_gpu == -1 || best_mem < free_memory ) {
            std::cerr << "updating best fit gpu to " << gpus.first << std::endl;
            best_gpu = gpus.first;
            best_mem = free_memory;
          }
        }
      }
    }

    //we didnt find one, so we need to reply retry
    if (best_gpu == -1) {
      std::cerr << "no gpus available for the request, retry." << std::endl;
      return -1;
    }
    //at this point we will be able to schedule
    (*gpu_states)[best_gpu].used_memory += requested_memory;

    for (auto& gpu_wks : (*workers)[best_gpu]) {
      if (gpu_wks.second.busy == false) {
        gpu_wks.second.busy = true;
        gpu_wks.second.used_memory = requested_memory;
        std::cerr << "scheduling worker at " << gpu_wks.first << std::endl;
        return gpu_wks.first;
      }
    }

  }
};

#endif