#ifndef __QNX__
/* Portable test transport, not a substitute for QNX scheduling or IPC.
 * Loopback TCP, one producer; 32-byte network-order records, including float
 * bits. Partial records survive receive timeouts; EOF drops partial state. */
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET socket_t;
#define BAD_SOCKET INVALID_SOCKET
#define close_socket closesocket
#else
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
typedef int socket_t;
#define BAD_SOCKET (-1)
#define close_socket close
#endif
#include "ipc.h"
#include "runtime.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>

enum { WIRE_CUE = 1, WIRE_HEARTBEAT = 2, WIRE_BYTES = 32 };
struct ipc_server {
    socket_t listener, peer;
    unsigned char record[WIRE_BYTES];
    size_t used;
    int pending;
};
struct ipc_conn { socket_t sock; };

static int network_init(void)
{
#ifdef _WIN32
    static int started;
    WSADATA data;
    if (!started) {
        if (WSAStartup(MAKEWORD(2, 2), &data)) { errno = EIO; return -1; }
        started = 1;
    }
#else
    signal(SIGPIPE, SIG_IGN);
#endif
    return 0;
}
static int would_block(void)
{
#ifdef _WIN32
    int e = WSAGetLastError();
    return e == WSAEWOULDBLOCK || e == WSAEINTR;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR;
#endif
}
static int nonblocking(socket_t fd)
{
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));
#ifdef _WIN32
    { u_long mode = 1; return ioctlsocket(fd, FIONBIO, &mode); }
#else
    return fcntl(fd, F_SETFL, O_NONBLOCK);
#endif
}
static int ready(socket_t fd, int write_ready, int ms)
{
    fd_set fds;
    struct timeval tv = {ms / 1000, (ms % 1000) * 1000};
    FD_ZERO(&fds); FD_SET(fd, &fds);
    return select((int)fd + 1, write_ready ? NULL : &fds,
                  write_ready ? &fds : NULL, NULL, ms < 0 ? NULL : &tv);
}
static int transfer(socket_t fd, void *buffer, size_t bytes, int sending)
{
    size_t done = 0;
    while (done < bytes) {
        int n = sending ? send(fd, (const char *)buffer + done, (int)(bytes - done), 0)
                        : recv(fd, (char *)buffer + done, (int)(bytes - done), 0);
        if (n > 0) { done += (size_t)n; continue; }
        if (n < 0 && would_block()) {
            if (ready(fd, sending, -1) >= 0) continue;
        }
        errno = ECONNRESET; return -1;
    }
    return 0;
}
static void pack(uint32_t out[8], unsigned kind, const cue_msg_t *msg)
{
    uint32_t score;
    memset(out, 0, WIRE_BYTES);
    out[0] = htonl(kind);
    if (!msg) return;
    memcpy(&score, &msg->score, sizeof(score));
    out[1] = htonl(msg->type); out[2] = htonl(msg->dir);
    out[3] = htonl(score);
    out[4] = htonl((uint32_t)(msg->t_capture >> 32));
    out[5] = htonl((uint32_t)msg->t_capture);
    out[6] = htonl(msg->frame_id);
}
ipc_server_t *ipc_server_open(const char *name)
{
    ipc_server_t *s;
    struct sockaddr_in addr;
#ifndef _WIN32
    int one = 1;
#endif
    if (strcmp(name, CUE_CHANNEL_NAME)) { errno = EINVAL; return NULL; }
    if (network_init() < 0) return NULL;
    s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->peer = BAD_SOCKET;
    s->listener = socket(AF_INET, SOCK_STREAM, 0);
    if (s->listener == BAD_SOCKET) { free(s); return NULL; }
#ifdef _WIN32
    /* SO_REUSEADDR on Windows could allow two live servers on one port. */
    { int one = 1; setsockopt(s->listener, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                            (const char *)&one, sizeof(one)); }
#else
    setsockopt(s->listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET; addr.sin_port = htons(CUE_TCP_PORT);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(s->listener, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(s->listener, 4) != 0 || nonblocking(s->listener) != 0) {
        ipc_server_close(s); errno = EADDRINUSE; return NULL;
    }
    return s;
}
ipc_conn_t *ipc_client_open(const char *name)
{
    ipc_conn_t *c;
    struct sockaddr_in addr;
    if (strcmp(name, CUE_CHANNEL_NAME)) { errno = EINVAL; return NULL; }
    if (network_init() < 0) return NULL;
    c = malloc(sizeof(*c));
    if (!c) return NULL;
    c->sock = socket(AF_INET, SOCK_STREAM, 0);
    if (c->sock == BAD_SOCKET) { free(c); return NULL; }
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET; addr.sin_port = htons(CUE_TCP_PORT);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(c->sock, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        nonblocking(c->sock) != 0) {
        ipc_client_close(c); errno = ECONNREFUSED; return NULL;
    }
    return c;
}
int ipc_send_cue(ipc_conn_t *c, const cue_msg_t *msg)
{
    uint32_t record[8], ack;
    pack(record, WIRE_CUE, msg);
    if (transfer(c->sock, record, sizeof(record), 1) < 0 ||
        transfer(c->sock, &ack, sizeof(ack), 0) < 0) return -1;
    return (int32_t)ntohl(ack);
}
int ipc_send_heartbeat(ipc_conn_t *c)
{
    uint32_t record[8];
    int n;
    pack(record, WIRE_HEARTBEAT, NULL);
    n = send(c->sock, (const char *)record, sizeof(record), 0);
    if (n == sizeof(record)) return 0;
    if (n < 0 && would_block()) return 0; /* Drop, never wait. */
    /* A partial heartbeat must not corrupt the next record. */
    close_socket(c->sock); c->sock = BAD_SOCKET;
    errno = ECONNRESET; return -1;
}
static void drop_peer(ipc_server_t *s)
{
    if (s->peer != BAD_SOCKET) close_socket(s->peer);
    s->peer = BAD_SOCKET; s->used = 0; s->pending = 0;
}
int ipc_receive(ipc_server_t *s, cue_msg_t *msg, int *is_pulse, int timeout_ms)
{
    uint64_t deadline, now;
    if (timeout_ms < 0 || s->pending) { errno = EINVAL; return -1; }
    deadline = monotonic_ns() + (uint64_t)timeout_ms * 1000000;
    *is_pulse = 0;
    for (;;) {
        socket_t fd = s->peer == BAD_SOCKET ? s->listener : s->peer;
        int n, ms;
        now = monotonic_ns();
        ms = now < deadline ? (int)((deadline - now + 999999) / 1000000) : 0;
        n = ready(fd, 0, ms);
        if (n == 0) return IPC_TIMEOUT;
        if (n < 0) { if (would_block()) continue; errno = EIO; return -1; }
        if (s->peer == BAD_SOCKET) {
            s->peer = accept(s->listener, NULL, NULL);
            if (s->peer != BAD_SOCKET && nonblocking(s->peer) != 0) drop_peer(s);
        } else {
            n = recv(s->peer, (char *)s->record + s->used,
                     (int)(WIRE_BYTES - s->used), 0);
            if (n == 0 || (n < 0 && !would_block())) drop_peer(s);
            else if (n > 0) {
                uint32_t words[8], score;
                s->used += (size_t)n;
                if (s->used == WIRE_BYTES) {
                    memcpy(words, s->record, WIRE_BYTES); s->used = 0;
                    if (ntohl(words[0]) == WIRE_HEARTBEAT) {
                        *is_pulse = 1; return 0;
                    }
                    if (ntohl(words[0]) != WIRE_CUE ||
                        ntohl(words[1]) != CUE_MSG_TYPE || ntohl(words[2]) > CUE_FAULT) {
                        drop_peer(s); errno = EINVAL; return -1;
                    }
                    memset(msg, 0, sizeof(*msg));
                    msg->type = (uint16_t)ntohl(words[1]);
                    msg->dir = (uint8_t)ntohl(words[2]);
                    score = ntohl(words[3]); memcpy(&msg->score, &score, sizeof(score));
                    msg->t_capture = ((uint64_t)ntohl(words[4]) << 32) | ntohl(words[5]);
                    msg->frame_id = ntohl(words[6]); s->pending = 1;
                    return 1;
                }
            }
        }
        if (monotonic_ns() >= deadline) return IPC_TIMEOUT;
    }
}
int ipc_reply(ipc_server_t *s, int rcvid, int status)
{
    uint32_t ack = htonl((uint32_t)status);
    int result;
    if (rcvid != 1 || !s->pending) { errno = EINVAL; return -1; }
    /* Only four bytes; a slow/malicious test peer must not stall the watchdog. */
    result = send(s->peer, (const char *)&ack, sizeof(ack), 0);
    s->pending = 0;
    if (result != sizeof(ack)) { drop_peer(s); errno = ECONNRESET; return -1; }
    return 0;
}
void ipc_server_close(ipc_server_t *s)
{
    if (s) { drop_peer(s); close_socket(s->listener); free(s); }
}
void ipc_client_close(ipc_conn_t *c)
{
    if (c) { if (c->sock != BAD_SOCKET) close_socket(c->sock); free(c); }
}
#endif
