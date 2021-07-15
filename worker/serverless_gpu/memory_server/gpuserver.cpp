#include "gpuserver.hpp"

#include <map>
#include <memory>
#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include "cuda_helper.hpp"
#include "unix_socket.hpp"

namespace GPUMemoryServer {

#define BUF_SIZE 1024

//The default value for flags is cudaMemAttachGlobal. If cudaMemAttachGlobal is specified, then this memory is accessible from any stream on any device.
//cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags = cudaMemAttachGlobal ) 

//https://forums.developer.nvidia.com/t/gpu-inter-process-communications-ipc-question/35936/5

//when we need to enable peer access
//https://github.com/NVIDIA/cuda-samples/blob/master/Samples/simpleIPC/simpleIPC.cu

void Server::run() {
    //go to our gpu
    checkCudaErrors( cudaSetDevice(gpu) );

    int sock = create_unix_socket(unix_socket_path);
    int client_sock = get_connection(sock);
    printf("Memory server of GPU %d ready\n", gpu);

    char* buffer = new char[BUF_SIZE];
    Request req;
    Reply reply;
    while(1) {
        int len = recv_message(sock, buffer, BUF_SIZE);
        //TODO: better (de)serialization
        memcpy(&req, buffer, sizeof(Request));

        /*************************
         *   cudaMalloc request
         *************************/
        if (req.type == RequestType::ALLOC) {
            //allocate and get the ipc handle
            void *ptr;

            //TODO: if more than requested, use cudaMallocManaged
            //TODO: we need to check if cudaMallocManaged works with cudaIpcGetMemHandle 
            checkCudaErrors( cudaMalloc(&ptr, req.data.size) );
            cudaIpcMemHandle_t memHandle;
            checkCudaErrors( cudaIpcGetMemHandle(&memHandle, ptr) );

            //store on our metadata maps
            uint64_t key = (uint64_t) ptr;
            allocs[req.worker_id][key] = std::make_unique<Allocation>(req.data.size, ptr);

            //c++ wizardy:  https://stackoverflow.com/questions/2667355/mapint-int-default-values
            used_memory[req.worker_id] += req.data.size;

            //construct reply
            memcpy(&reply.memHandle, &memHandle, sizeof(cudaIpcMemHandle_t));
            memcpy(buffer, &reply, sizeof(Reply));
            //reply
            send_message(client_sock, buffer, sizeof(Reply));
        }
        /*************************
         *       cudaFree
         *************************/
        else if (req.type == RequestType::FREE) {
            uint64_t ptr = (uint64_t) req.data.devPtr;
            auto alloc = allocs[req.worker_id].find(ptr);

            //not found, illegal free
            if (alloc == allocs[req.worker_id].end()) {
                printf("Worker %d tried to free %p, which wasn't allocated\n", req.worker_id, req.data.devPtr);
            }
            else {
                //update used memory
                used_memory[req.worker_id] -= alloc->second->size;
                //this frees the handle and memory
                allocs[req.worker_id].erase(alloc);
            }
        }
        /*************************
         * worker finished, cleanup
         *************************/
        else if (req.type == RequestType::FINISHED) {
            used_memory[req.worker_id] = 0;
            allocs[req.worker_id].clear();
        }
    }
}

Allocation::~Allocation() {
    checkCudaErrors( cudaIpcCloseMemHandle(devPtr) );  //TODO: double check if we are opening for all requests
    checkCudaErrors( cudaFree(devPtr) );
}

//end of namespace
};

