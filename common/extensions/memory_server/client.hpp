#ifndef __MEMORYSERVER_CLIENT_HPP__
#define __MEMORYSERVER_CLIENT_HPP__

#include <stdint.h>
#include <string>
#include <zmq.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <memory>
#include <absl/containers/flat_hash_map.h>
#include "common.hpp"

#ifdef __cplusplus
extern "C" {
#endif

CUresult __internal_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
cudaError_t __internal_cudaMalloc(void **devPtr, size_t size);
cudaError_t __internal_cudaFree(void *devPtr);
void __internal_kernelIn();
void __internal_kernelOut();

#ifdef __cplusplus
}
#endif

namespace GPUMemoryServer {

    class Client {
        public:
        //zmq and comms stuff
        void* context;
        void* socket;
        char* buffer;
        bool memoryserver_mode;

        //device management
        int current_device, og_device;
        std::string uuid;
        //local mallocs
        struct LocalAlloc {
            void* devPtr;
            LocalAlloc(void* ptr) : devPtr(ptr) {}
            ~LocalAlloc() {
                cudaFree(devPtr);
            }
        };
        std::vector<std::unique_ptr<LocalAlloc>> local_allocs;
        
        //pointer translation
        absl::flat_hash_map<uint64_t, uint64_t> pointer_map;

        //simple functions
        void setMemoryServerMode(bool f) {
            memoryserver_mode = f;
        }
        bool isMemoryServerMode() {
            return memoryserver_mode;
        }
        void setUuid(std::string id) {
            uuid = id;
        }
        cudaError_t localMalloc(void** devPtr, size_t size);
        void cleanup();
        void kernelIn();
        void kernelOut();
        void setOriginalGPU();
        void setCurrentGPU(int id);
        int connectToGPU(uint16_t gpuId);
        Reply sendMallocRequest(uint64_t size);
        Reply sendFreeRequest(void* devPtr);
        Reply sendCleanupRequest();
        Reply sendMemoryRequestedValue(uint64_t mem_mb);

        static Client& getInstance() {
            static Client instance;
            return instance;
        }

        private:
        Reply sendRequest(Request &req);

        Client() {
            buffer = new char[BUF_SIZE];
            memoryserver_mode = false;
            og_device = -1;
        }
        ~Client();

        Client(Client const&)         = delete;
        void operator=(Client const&) = delete;
    };
}

#endif