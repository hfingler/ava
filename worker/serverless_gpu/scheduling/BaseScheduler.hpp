#ifndef __BASE_SCHEDULER_HPP__
#define __BASE_SCHEDULER_HPP__

#include <stdint.h>

// none of this interface/fields is set in stone, feel free to change
/* TODO features:
 *  - keep list of all gpus with their resources
 *  - make request accept some parameters
 */
class BaseScheduler {
 public:
  uint16_t ngpus;

  BaseScheduler(uint16_t ngpus) : ngpus(ngpus){};
  virtual int getGPU() = 0;
};

#endif