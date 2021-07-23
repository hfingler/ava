#ifndef __MEMORYSERVER_CLIENT_HPP__
#define __MEMORYSERVER_CLIENT_HPP__

#include <stdint.h>
#include <string>
#include "common.hpp"
#include <zmq.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t __internal_cudaMalloc(void **devPtr, size_t size);
cudaError_t __internal_cudaFree(void *devPtr);

#ifdef __cplusplus
}
#endif

namespace GPUMemoryServer {

    class Client {
        public:
        void* context;
        void* socket;
        char* buffer;
        bool is_connected;
        int current_device, og_device;
        std::string uuid;
        //for local mallocs
        struct LocalAlloc {
            void* devPtr;
            LocalAlloc(void* ptr) : devPtr(ptr) {}
            ~LocalAlloc() {
                cudaFree(devPtr);
            }
        };
        std::vector<std::unique_ptr<LocalAlloc>> local_allocs;

        inline bool isConnected() {
            return is_connected;
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
            is_connected = false;
            og_device = -1;
        }
        ~Client();

        Client(Client const&)         = delete;
        void operator=(Client const&) = delete;
    };
}

#endif