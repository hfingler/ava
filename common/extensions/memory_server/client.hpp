#ifndef __MEMORYSERVER_CLIENT_HPP__
#define __MEMORYSERVER_CLIENT_HPP__

#include <stdint.h>
#include <string>
#include "common.hpp"
#include <zmq.h>


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
        uint64_t uuid;

        inline bool isConnected() {
            return is_connected;
        }
        void setUuid(uint64_t id) {
            uuid = id;
        }
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
        }
        ~Client() {
            zmq_close(socket);
            //zmq_ctx_destroy(context);
        }

        Client(Client const&)         = delete;
        void operator=(Client const&) = delete;
    };
}

#endif