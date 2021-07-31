#ifndef __MEMORYSERVER_CLIENT_HPP__
#define __MEMORYSERVER_CLIENT_HPP__

#include <stdint.h>
#include <string>
#include <zmq.h>
#include <map>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <memory>
//#include <absl/containers/flat_hash_map.h>
#include <unordered_map>
#include <mutex>
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

uint32_t __internal_getCurrentDevice(); 
int32_t __internal_getDeviceCount();
void __internal_setDeviceCount(uint32_t dc);
void* __translate_ptr(void*);
const void* __translate_ptr(const void*);

namespace GPUMemoryServer {

    class Client {
        public:
        //zmq and comms stuff
        void* context;
        void* sockets[4];
        char* buffer;

        //let's keep one Reply here since we can always reuse it
        Reply reply;
        Request req;

        //device management
        int current_device, og_device;
        int device_count;
        std::string uuid;
        //local mallocs
        struct LocalAlloc {
            void* devPtr;
            uint64_t size;
            uint32_t device_id;
            LocalAlloc(void* ptr, uint64_t sz, uint32_t gpu) 
                : devPtr(ptr), size(sz), device_id(gpu) {}
            ~LocalAlloc() {
                printf("cudaFree on GPU [%u]  %p\n", device_id, devPtr);
                cudaFree(devPtr);
            }
        };
        std::map<uint64_t, std::unique_ptr<LocalAlloc>> local_allocs;

        //pointer translation
        //absl::flat_hash_map<uint64_t, void*> pointer_map;
        std::unordered_map<uint64_t, void*> pointer_map;
        void* translate_ptr(void* ptr);

        //migration
        void migrateToGPU(uint32_t new_gpuid, Migration migration_type);

        void setUuid(std::string id) {
            uuid = id;
        }
        cudaError_t localMalloc(void** devPtr, size_t size);
        cudaError_t localFree(void* devPtr);
        void cleanup();
        void kernelIn();
        void kernelOut();
        void setOriginalGPU();
        void setCurrentGPU(int id);
        void connectToGPU(uint32_t gpuid);
        void sendMallocRequest(uint64_t size);
        void sendFreeRequest(uint64_t size);
        void sendCleanupRequest();
        void sendMemoryRequestedValue(uint64_t mem_mb);

        static Client& getInstance() {
            static Client instance;
            return instance;
        }

        private:
        void sendRequest(Request &req);
        void handleReply();

        Client();
        ~Client();

        Client(Client const&)         = delete;
        void operator=(Client const&) = delete;
    };
}

#endif
