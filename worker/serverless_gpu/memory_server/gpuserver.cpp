#include "gpuserver.hpp"
#include <map>
#include <memory>
#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "cuda_helper.hpp"
#include <zmq.h>

namespace GPUMemoryServer {

#define BUF_SIZE 4096

//The default value for flags is cudaMemAttachGlobal. If cudaMemAttachGlobal is specified, then this memory is accessible from any stream on any device.
//cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags = cudaMemAttachGlobal ) 

//https://forums.developer.nvidia.com/t/gpu-inter-process-communications-ipc-question/35936/5

//when we need to enable peer access
//https://github.com/NVIDIA/cuda-samples/blob/master/Samples/simpleIPC/simpleIPC.cu

void Server::run() {
    //create zmq listener
    void *context = zmq_ctx_new();
    void *responder = zmq_socket(context, ZMQ_REP);
    std::string addr = "ipc://";
    addr += unix_socket_path;
    int rc = zmq_bind(responder, addr.c_str());

    //go to our gpu
    checkCudaErrors( cudaSetDevice(gpu) );

    char* buffer = new char[BUF_SIZE];
    printf("Memory server of GPU %d ready\n", gpu);
    Reply rep;
    while(1) {
        zmq_recv(responder, buffer, BUF_SIZE, 0);
        handleRequest(buffer, &rep);
        memcpy(buffer, &rep, sizeof(Reply));
        zmq_send(responder, buffer, sizeof(Reply), 0);
    }
}

void Server::handleRequest(char* buffer, Reply* rep) {
    Request req;

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
        memcpy(&(rep->memHandle), &memHandle, sizeof(cudaIpcMemHandle_t));
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

        //construct reply
        memcpy(&(rep->memHandle), 0, sizeof(Reply));
    }
    /*************************
     * worker finished, cleanup
     *************************/
    else if (req.type == RequestType::FINISHED) {
        used_memory[req.worker_id] = 0;
        allocs[req.worker_id].clear();
        //construct reply
        memcpy(&(rep->memHandle), 0, sizeof(Reply));
    }
}   

Allocation::~Allocation() {
    checkCudaErrors( cudaIpcCloseMemHandle(devPtr) );  //TODO: double check if we are opening for all requests
    checkCudaErrors( cudaFree(devPtr) );
}

//end of namespace
};

