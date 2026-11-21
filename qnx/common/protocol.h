#ifndef DODGE_PROTOCOL_H
#define DODGE_PROTOCOL_H
#include <stdint.h>

/* Local IPC ABI: all processes must use this header and the same build/ABI.
 * Values deliberately follow CODEX_TASK_QNX.md, not the older draft enum. */
#define CUE_CHANNEL_NAME "blindfolded_dodge_cue"
#define CUE_TCP_PORT 45873 /* loopback only; portable test backend */
#define CUE_MSG_TYPE 0x600 /* above QNX's reserved _IO_MAX range */
#ifdef __QNX__
#include <sys/neutrino.h>
#define CUE_HEARTBEAT_CODE _PULSE_CODE_MINAVAIL
#else
#define CUE_HEARTBEAT_CODE 1
#endif

enum cue_dir {
    CUE_NONE = 0, CUE_INCOMING = 1, CUE_LEFT = 2,
    CUE_RIGHT = 3, CUE_DUCK = 4, CUE_FAULT = 5
};
typedef struct {
    uint16_t type;
    uint8_t dir;
    float score;
    uint64_t t_capture; /* CLOCK_MONOTONIC nanoseconds, same host */
    uint32_t frame_id;
} cue_msg_t;

static inline const char *cue_name(unsigned dir)
{
    static const char *const names[] = {
        "NONE", "INCOMING", "LEFT", "RIGHT", "DUCK", "FAULT"
    };
    return dir <= CUE_FAULT ? names[dir] : "INVALID";
}
#endif
