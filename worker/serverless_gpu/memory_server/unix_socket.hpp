#ifndef __MEMORYSERVER_UNIXSOCKET_HPP__
#define __MEMORYSERVER_UNIXSOCKET_HPP__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <unistd.h>

//from https://www.ibm.com/docs/en/ztpf/2020?topic=considerations-unix-domain-sockets
int create_unix_socket(std::string unix_socket_path) {    
    int server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_sock == -1){
        printf("SOCKET ERROR = %s", strerror(errno));
        exit(1);
    }
    
    struct sockaddr_un server_sockaddr;
    memset(&server_sockaddr, 0, sizeof(struct sockaddr_un));
    server_sockaddr.sun_family = AF_UNIX;
    strcpy(server_sockaddr.sun_path, unix_socket_path.c_str()); 
    int len = sizeof(server_sockaddr);

    unlink(unix_socket_path.c_str());
    int rc = bind(server_sock, (struct sockaddr *) &server_sockaddr, len);
    if (rc == -1){
        printf("BIND ERROR = %s", strerror(errno));
        close(server_sock);
        exit(1);
    }

    rc = listen(server_sock, 5);
    if (rc == -1){ 
        printf("LISTEN ERROR: %s\n", strerror(errno));
        close(server_sock);
        exit(1);
    }

    return server_sock;
}

int get_connection(int socket) {
    int csocket = accept(socket, 0, 0);
    if (csocket == -1){ 
        printf("LISTEN ERROR: %s\n", strerror(errno));
        close(socket);
        exit(1);
    }
    return csocket;
}

int recv_message(int socket, char* buf, uint32_t len) {
    int bytes_rec = recv(socket, buf, len, 0);
    if (bytes_rec == -1){
        printf("RECVFROM ERROR = %s", strerror(errno));
        close(socket);
        exit(1);
    }
    return bytes_rec;
}

int send_message(int client_sock, char* buf, uint32_t len) {
    int rc = send(client_sock, buf, strlen(buf), 0);
    if (rc == -1) {
        printf("SEND ERROR: %s", strerror(errno));
        close(client_sock);
        exit(1);
    }   
}


#endif