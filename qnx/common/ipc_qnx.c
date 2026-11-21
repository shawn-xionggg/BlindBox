#ifdef __QNX__
#include "ipc.h"
#include "runtime.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/dispatch.h>
#include <sys/iomsg.h>

struct ipc_server { name_attach_t *attach; };
struct ipc_conn { int coid; };

ipc_server_t *ipc_server_open(const char *name)
{
    ipc_server_t *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->attach = name_attach(NULL, name, 0);
    if (!s->attach) { free(s); return NULL; }
    return s;
}
ipc_conn_t *ipc_client_open(const char *name)
{
    ipc_conn_t *c = malloc(sizeof(*c));
    if (!c) return NULL;
    c->coid = name_open(name, 0);
    if (c->coid < 0) { free(c); return NULL; }
    return c;
}
int ipc_send_cue(ipc_conn_t *c, const cue_msg_t *msg)
{
    int status = 0;
    /* Native microkernel message passing: send-blocking and reply-blocking
     * impose back-pressure instead of accumulating stale cue commands. */
    if (MsgSend(c->coid, msg, sizeof(*msg), &status, sizeof(status)) < 0)
        return -1;
    return status;
}
int ipc_send_heartbeat(ipc_conn_t *c)
{
    /* A pulse is queued without blocking for a receiver/reply. */
    return MsgSendPulse(c->coid, -1, CUE_HEARTBEAT_CODE, 0);
}
int ipc_receive(ipc_server_t *s, cue_msg_t *msg, int *is_pulse, int timeout_ms)
{
    union { struct _pulse pulse; cue_msg_t cue; uint16_t type; } in;
    struct _msg_info info;
    uint64_t deadline, now, remaining;
    int rcvid;
    if (timeout_ms < 0) { errno = EINVAL; return -1; }
    *is_pulse = 0;
    deadline = monotonic_ns() + (uint64_t)timeout_ms * 1000000;
    for (;;) {
        now = monotonic_ns();
        remaining = now < deadline ? deadline - now : 0;
        /* Relative CLOCK_MONOTONIC timeout immediately before MsgReceive.
         * Preserve the deadline while consuming connection/system traffic. */
        if (TimerTimeout(CLOCK_MONOTONIC, _NTO_TIMEOUT_RECEIVE, NULL,
                         &remaining, NULL) < 0) return -1;
        memset(&in, 0, sizeof(in));
        rcvid = MsgReceive(s->attach->chid, &in, sizeof(in), &info);
        if (rcvid < 0) return errno == ETIMEDOUT ? IPC_TIMEOUT : -1;
        if (rcvid == 0) {
            if (in.pulse.code == CUE_HEARTBEAT_CODE) {
                *is_pulse = 1; return 0; /* Never reply to pulses. */
            }
            if (in.pulse.code == _PULSE_CODE_DISCONNECT)
                ConnectDetach(in.pulse.scoid);
            else if (in.pulse.code == _PULSE_CODE_UNBLOCK)
                MsgError(in.pulse.value.sival_int, EINTR);
        } else if (in.type == _IO_CONNECT) {
            MsgReply(rcvid, EOK, NULL, 0); /* name_open handshake */
        } else if (in.type <= _IO_MAX) {
            MsgError(rcvid, ENOSYS);
        } else if ((size_t)info.msglen != sizeof(cue_msg_t) ||
                   (size_t)info.srcmsglen != sizeof(cue_msg_t) ||
                   in.cue.type != CUE_MSG_TYPE) {
            MsgError(rcvid, EINVAL);
        } else {
            *msg = in.cue; return rcvid;
        }
        if (monotonic_ns() >= deadline) return IPC_TIMEOUT;
    }
}
int ipc_reply(ipc_server_t *s, int rcvid, int status)
{
    (void)s;
    return MsgReply(rcvid, EOK, &status, sizeof(status));
}
void ipc_server_close(ipc_server_t *s)
{
    if (s) { name_detach(s->attach, 0); free(s); }
}
void ipc_client_close(ipc_conn_t *c)
{
    if (c) { name_close(c->coid); free(c); }
}
#endif
