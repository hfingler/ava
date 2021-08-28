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

bool __internal_allContextsEnabled(); 
bool __internal_setAllContextsEnabled(bool f);
uint32_t __internal_getCurrentDevice(); 
int32_t __internal_getDeviceCount();
void __internal_setDeviceCount(uint32_t dc);
void* __translate_ptr(void*);
const void* __translate_ptr(const void*);
cudaStream_t __translate_stream(cudaStream_t key);
cudaEvent_t __translate_event(cudaEvent_t key);

void __cmd_handle_in();
void __cmd_handle_out();

namespace GPUMemoryServer {

    class Client {
        public:
        //zmq and comms stuff
        void* context;
        void* central_socket;
        std::mutex sockmtx;
        bool enable_reporting;

        //device management
        int current_device, og_device;
        int device_count;
        Migration migrated_type;
        uint32_t listen_port;

        std::string uuid;
        //local mallocs
        struct LocalAlloc {
            void* devPtr;
            uint64_t size;
            uint32_t device_id;
            LocalAlloc(void* ptr, uint64_t sz, uint32_t gpu) 
                : devPtr(ptr), size(sz), device_id(gpu) {}
            ~LocalAlloc() {
#ifndef NDEBUG
                printf("cudaFree on GPU [%u]  %p\n", device_id, devPtr);
#endif
                cudaFree(devPtr);
            }
        };
        //std::map<uint64_t, std::unique_ptr<LocalAlloc>> local_allocs;
        std::vector<std::unique_ptr<LocalAlloc>> local_allocs;

        //pointer translation
        struct DevPointerTranslate {
            uint64_t size;
            uint64_t dstPtr;
        };
        std::map<uint64_t, DevPointerTranslate> pointer_map;
        void* translateDevicePointer(void* ptr);
        void tryRemoveFromPointerMap(void* ptr);
        bool isInPointerMap(void* ptr);
        bool pointerIsMapped(void* ptr, int32_t gpuid = -1);

        //stream translation
        std::map<cudaStream_t, std::map<uint32_t,cudaStream_t>> streams_map;
        //event translation
        std::map<cudaEvent_t, std::map<uint32_t,cudaEvent_t>> events_map;

        //migration
        void migrateToGPU(uint32_t new_gpuid, Migration migration_type);

        void setListenPort(uint32_t port) {
            listen_port = port;
        }
        void setUuid(std::string id) {
            uuid = id;
        }
        cudaError_t localMalloc(void** devPtr, size_t size);
        cudaError_t localFree(void* devPtr);
        void cleanup(uint32_t cd);
        void notifyReady();
        void fullCleanup();
        void kernelIn();
        void kernelOut();
        void resetCurrentGPU();
        void setCurrentGPU(int id);
        void connectToCentral();
        void reportMalloc(uint64_t size);
        void reportFree(uint64_t size);
        void reportCleanup(uint32_t gpuid);
        void reportMemoryRequested(uint64_t mem_mb);

        static Client& getInstance() {
            static Client instance;
            return instance;
        }

        private:
        void sendRequest(Request &req);
        void handleReply(Reply& reply);

        Client();
        ~Client();

        Client(Client const&)         = delete;
        void operator=(Client const&) = delete;
    };
}

#endif
