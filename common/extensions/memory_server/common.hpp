#ifndef __MEMORYSERVER_COMMON_HPP__
#define __MEMORYSERVER_COMMON_HPP__

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <string.h>

#define BUF_SIZE 4096

namespace GPUMemoryServer {

    inline std::string get_base_socket_path() {
        return std::string("ipc:///tmp/gpumemserver_sock");
    }

    enum RequestType { ALLOC, FREE, FINISHED, MEMREQUESTED };

    union RequestData {
        uint64_t size;
        void* devPtr;
    };

    struct Request {
        RequestType type;
        uint64_t worker_id;
        RequestData data;
    };

    union ReplyData {
        cudaIpcMemHandle_t memHandle;
    };

    struct Reply {
        cudaError_t returnErr;
        ReplyData data;
    };
}

#endif