#include "gpuserver.hpp"

#include <map>
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

//TODO: we need to check if cudaMallocManaged works with cudaIpcGetMemHandle 

void Server::run() {
    //go to our gpu
    checkCudaErrors( cudaSetDevice(gpu) );

    int sock = create_unix_socket(unix_socket_path);
    int client_sock = get_connection(sock);
    printf("Memory server of GPU %d ready\n", gpu);

    //TODO: we need to keep a map of all allocations we did
    std::map<uint64_t, Allocation> allocs;

    char* buffer = new char[BUF_SIZE];
    Request req;
    Reply reply;
    while(1) {
        int len = recv_message(sock, buffer, BUF_SIZE);

        //TODO: better (de)serialization
        memcpy(&req, buffer, sizeof(Request));

        if (req.type == RequestType::ALLOC) {
            void *ptr;
            checkCudaErrors( cudaMalloc(&ptr, req.size) );

            cudaIpcMemHandle_t memHandle;
            checkCudaErrors( cudaIpcGetMemHandle(&memHandle, ptr) );
            //TODO: save handle on a map, so we can cudaIpcCloseMemHandle when it's freed

            //construct reply
            memcpy(&reply.memHandle, &memHandle, sizeof(cudaIpcMemHandle_t));
            memcpy(buffer, &reply, sizeof(Reply));
            //reply
            send_message(client_sock, buffer, sizeof(Reply));
        }
        else if (req.type == RequestType::FREE) {
            //TODO
        }
    }
}

//end of namespace
};

