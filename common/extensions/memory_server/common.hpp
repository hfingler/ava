#ifndef __MEMORYSERVER_COMMON_HPP__
#define __MEMORYSERVER_COMMON_HPP__

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <string.h>
#include <cstddef>

#define BUF_SIZE 16384

namespace GPUMemoryServer {

    inline std::string get_base_socket_path() {
        return std::string("ipc:///tmp/gpumemserver_sock");
    }

    enum RequestType { ALLOC, FREE, FINISHED, MEMREQUESTED,
        KERNEL_IN, KERNEL_OUT,
        FETCH_ALL_PTRS, MIGRATE};

    union RequestData {
        uint64_t size;
        void* devPtr;
    };

    struct Request {
        RequestType type;
        RequestData data;
        char worker_id[40]; //36 is the size of a uuid v4
    };
    union ReplyData {
        cudaIpcMemHandle_t memHandle;
        char migrate;
    };

    struct Reply {
        cudaError_t returnErr;
        ReplyData data;
        char guard;
        //pad until BUF_SIZE, maybe offsetof works here?
        char piggyback[BUF_SIZE-2-sizeof(cudaError_t)-sizeof(ReplyData)];
    };

    struct MigrationHeader {
        uint32_t n_buffers;  //amount of buffers that are piggybacked by this struct
    };

    struct PointerPair {
        uint64_t ptr;
        uint64_t size;
    };
}

#endif