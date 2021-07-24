#ifndef __MEMORYSERVER_CLIENT_HPP__
#define __MEMORYSERVER_CLIENT_HPP__

#include <stdint.h>
#include <string>
#include <zmq.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <memory>
//#include <absl/containers/flat_hash_map.h>
#include <unordered_map>
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
void* __translate_ptr(void*);

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

        //let's keep one Reply here since we can always reuse it
        //it's 16K so we dont wanna keep creating these
        Reply reply;
        Request req;

        //this sucks
        Reply* getReply() {
            return &reply;
        }

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
        //absl::flat_hash_map<uint64_t, void*> pointer_map;
        std::unordered_map<uint64_t, void*> pointer_map;
        void* translate_ptr(void* ptr);

        //migration
        void sendGetAllPointersRequest();
        void migrateToGPU(uint32_t new_gpuid);

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
        void sendMallocRequest(uint64_t size);
        void sendFreeRequest(void* devPtr);
        void sendCleanupRequest();
        void sendMemoryRequestedValue(uint64_t mem_mb);

        static Client& getInstance() {
            static Client instance;
            return instance;
        }

        private:
        void sendRequest(Request &req, uint32_t size = offsetof(Request, guard), void* sock = 0);

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