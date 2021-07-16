#include <stdint.h>
#include <string>
#include <sstream>
#include <string.h>
#include <stdlib.h>
#include "common.hpp"
#include "client.hpp"
#include <zmq.h>

namespace GPUMemoryServer {

    void Client::connectToGPU(uint16_t gpuId) {
        context = zmq_ctx_new();
        socket = zmq_socket(context, ZMQ_REQ);
        std::ostringstream stringStream;
        stringStream << get_base_socket_path() << gpuId;
        socket = zmq_connect(socket, stringStream.str().c_str());
    }

    void* Client::sendMallocRequest(uint64_t size) {
        Request req;
        req.type = RequestType::ALLOC;
        req.data.size = size;

        Reply rep = sendRequest(req);
        return rep.data.devPtr;
    }
    
    void Client::sendFreeRequest(void* devPtr) {
        Request req;
        req.type = RequestType::FREE;
        req.data.devPtr = devPtr;

        sendRequest(req);
    }
    
    //this is called when we are done, so cleanup too
    void Client::sendCleanupRequest() {
        Request req;
        req.type = RequestType::FINISHED;
        sendRequest(req);
    }

    Reply Client::sendRequest(Request &req) {
        memcpy(buffer, &req, sizeof(Request));
        zmq_send(socket, buffer, sizeof(Request), 0);
        zmq_recv(socket, buffer, sizeof(Reply), 0);

        Reply rep;
        memcpy(&rep, buffer, sizeof(Reply));
        return rep;
    }
}