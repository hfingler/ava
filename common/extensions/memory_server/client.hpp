#ifndef __MEMORYSERVER_CLIENT_HPP__
#define __MEMORYSERVER_CLIENT_HPP__

#include <stdint.h>
#include <string>
#include <zmq.h>
#include "common.hpp"



namespace GPUMemoryServer {

    class Client {
        public:
        void* context;
        void* socket;
        char* buffer;

        void connectToGPU(uint16_t gpuId);
        void* sendMallocRequest(uint64_t size);
        void sendFreeRequest(void* devPtr);
        void sendCleanupRequest();

        static Client& getInstance() {
            static Client instance;
            return instance;
        }

        private:
        Reply sendRequest(Request &req);

        Client() {
            buffer = new char[BUF_SIZE];
        }
        ~Client() {
            zmq_close(socket);
            zmq_ctx_destroy(context);
        }

        Client(Client const&)         = delete;
        void operator=(Client const&) = delete;
    };
}

#endif