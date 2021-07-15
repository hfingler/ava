#ifndef __MEMORYSERVER_UNIXSOCKET_HPP__
#define __MEMORYSERVER_UNIXSOCKET_HPP__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <errno.h>
#include <unistd.h>

struct fd_set master_set, working_set;
int max_sd;

//from https://www.ibm.com/docs/en/ztpf/2020?topic=considerations-unix-domain-sockets
int create_unix_socket(std::string unix_socket_path) {    
    int server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_sock == -1){
        printf("SOCKET ERROR = %s", strerror(errno));
        exit(1);
    }
    
    int on = 1;
    rc = ioctl(server_sock, FIONBIO, (char*)&on);
    if (rc < 0) {
        printf("SOCKET ERROR = %s", strerror(errno));
        close(server_sock);
        exit(-1);
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

    rc = listen(server_sock, 32);
    if (rc == -1){ 
        printf("LISTEN ERROR: %s\n", strerror(errno));
        close(server_sock);
        exit(1);
    }

    FD_ZERO(&master_set);
    max_sd = server_sock;
    FD_SET(server_sock, &master_set);

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

int recv_message(int listen_sock, char* buf, uint32_t len, uint32_t expected) {
    int rc;
    while (1) {
        memcpy(&working_set, &master_set, sizeof(master_set));
        rc = select(max_sd + 1, &working_set, NULL, NULL, NULL);
        if (rc < 0) {
            printf("SELECT ERROR: %s\n", strerror(errno));
            exit(1);
        }

        int desc_ready = rc;
        for (int i = 0; i <= listen_sock  &&  desc_ready > 0; i++) {
            if (FD_ISSET(i, &working_set))
            {
                desc_ready -= 1;
                //incoming connection request
                if (i == listen_sock) {
                    int client_sock;
                    while (1) {
                        client_sock = accept(listen_sd, NULL, NULL);
                        //if we are done
                        if (client_sock < 0) {
                            if (errno != EWOULDBLOCK) {
                                perror("  accept() failed");
                                exit(1);
                            }
                            break;
                        }

                        printf("New worker connection - %d\n", client_sock);
                        FD_SET(client_sock, &master_set);
                        if (client_sock > max_sd) {
                            max_sd = client_sock;
                        }
                    }
                }
                //incoming data
                else {
                    bool closeConn = false;
                    uint32_t total_recvd = 0;
                    //this assumes that we only receive one message at a time from a socket
                    //which is sorta acceptable since it's a request/reply scenario
                    while (total_recvd != expected) {
                        rc = recv(i, buf+total_recvd, len-total_recvd, 0);
                        if (rc < 0) {
                            if (errno != EWOULDBLOCK) {
                                perror("  recv() failed");
                                exit(1);
                            }
                        }
                        if (rc == 0) {
                            printf("Worker connection closed\n");
                            closeConn = true;
                            break;
                        }

                        total_recvd += rc;
                        if (total_recvd != expected) {
                           printf("Didn't receive full message, looping until we do\n"); 
                        }
                    }


                    if (closeConn) {

                        
                    }


                }
            }
        }
    }



}



int bytes_rec = recv(socket, buf, len, 0);
    if (bytes_rec == -1){
        printf("RECVFROM ERROR = %s", strerror(errno));
        close(socket);
        exit(1);
    }
    return bytes_rec;


int send_message(int client_sock, char* buf, uint32_t len) {
    int rc = send(client_sock, buf, strlen(buf), 0);
    if (rc == -1) {
        printf("SEND ERROR: %s", strerror(errno));
        close(client_sock);
        exit(1);
    }   
}


#endif