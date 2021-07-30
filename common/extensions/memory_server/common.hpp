#ifndef __MEMORYSERVER_COMMON_HPP__
#define __MEMORYSERVER_COMMON_HPP__

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <string.h>
#include <cstddef>

#define BUF_SIZE 4094

namespace GPUMemoryServer {

    inline std::string get_base_socket_path() {
        return std::string("ipc:///tmp/gpumemserver_sock_");
    }

    enum RequestType { ALLOC, FREE, FINISHED, MEMREQUESTED,
        KERNEL_IN, KERNEL_OUT};

    enum Migration { NOPE, MEMORY, KERNEL};

    union RequestData {
        uint64_t size;
    };

    struct Request {
        RequestType type;
        RequestData data;
        char worker_id[40]; //36 is the size of a uuid v4
    };

    struct Reply {
        Migration migrate;
        uint32_t target_device;
    };

}

#endif