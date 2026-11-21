#ifndef DODGE_IPC_H
#define DODGE_IPC_H
#include "protocol.h"
typedef struct ipc_server ipc_server_t;
typedef struct ipc_conn ipc_conn_t;
#define IPC_TIMEOUT (-2)
ipc_server_t *ipc_server_open(const char *name);
ipc_conn_t *ipc_client_open(const char *name);
/* Send blocks until server replies; returns application status, or -1 on error.
 * One thread per connection; there may be only one outstanding cue. */
int ipc_send_cue(ipc_conn_t *conn, const cue_msg_t *msg);
/* >0: rcvid needing exactly one reply; 0: heartbeat (is_pulse=1);
 * IPC_TIMEOUT: deadline expired; -1: error. Internal QNX pulses are consumed.
 * timeout_ms >=0, including 0 for polling. */
int ipc_receive(ipc_server_t *server, cue_msg_t *msg, int *is_pulse,
                int timeout_ms);
int ipc_reply(ipc_server_t *server, int rcvid, int status);
/* Never waits for a server reply. Portable backend may drop a heartbeat if
 * its socket is full; a broken transport is reported as -1. */
int ipc_send_heartbeat(ipc_conn_t *conn);
void ipc_server_close(ipc_server_t *server);
void ipc_client_close(ipc_conn_t *conn);
#endif
