#include "ipc.h"
#include "runtime.h"
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>

static volatile sig_atomic_t stopping;
static const unsigned *pattern;
static unsigned step;
static uint64_t transition;
static int output_on;
static void stop_handler(int sig) { (void)sig; stopping = 1; }

static void buzzer_output(int on)
{
    output_on = on;
#ifdef __QNX__
    /* TODO: verify on hardware. Pi 5 GPIO lives behind RP1. No verified
     * QNX resource-manager API/pin mapping is available for this board.
     * Replace this stub with a validated GPIO write; do not guess MMIO.
     * Presently QNX logs patterns but DOES NOT energize a physical buzzer. */
    printf("BUZZER gpio_stub=%s\n", on ? "on" : "off");
#else
    printf("BUZZER stdout=%s\n", on ? "on" : "off");
#endif
}
static void buzzer_on(unsigned ms)
{
    buzzer_output(1);
    transition = monotonic_ns() + (uint64_t)ms * 1000000;
}
static void buzzer_pattern(unsigned dir)
{
    /* Alternating on/off durations; zero terminates. New cues preempt old
     * patterns so their first buzz never queues behind a previous cue. */
    static const unsigned incoming[] = {60, 0};
    static const unsigned left[] = {40, 45, 40, 0};
    static const unsigned right[] = {130, 0};
    static const unsigned duck[] = {35, 35, 35, 35, 35, 0};
    static const unsigned fault[] = {180, 80, 180, 0};
    pattern = NULL; step = 0;
    if (output_on) buzzer_output(0);
    switch (dir) {
    case CUE_INCOMING: pattern = incoming; break;
    case CUE_LEFT: pattern = left; break;
    case CUE_RIGHT: pattern = right; break;
    case CUE_DUCK: pattern = duck; break;
    case CUE_FAULT: pattern = fault; break;
    default: break;
    }
    printf("BUZZER pattern=%s\n", cue_name(dir));
    if (pattern) buzzer_on(pattern[0]);
}
static void buzzer_tick(uint64_t now)
{
    if (!pattern || now < transition) return;
    ++step;
    if (!pattern[step]) {
        if (output_on) buzzer_output(0);
        pattern = NULL;
    } else if (step % 2 == 0) buzzer_on(pattern[step]);
    else {
        buzzer_output(0);
        transition = now + (uint64_t)pattern[step] * 1000000;
    }
}
int main(int argc, char **argv)
{
    ipc_server_t *server;
    uint64_t start, last_heartbeat;
    double run_for = 0;
    int faulted = 0;
    if (argc == 3 && strcmp(argv[1], "--run-for") == 0) {
        char *end;
        run_for = strtod(argv[2], &end);
        if (*end || !(run_for > 0) || run_for > 86400) return 2;
    } else if (argc != 1) {
        fprintf(stderr, "usage: cue_driver [--run-for seconds]\n"); return 2;
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGINT, stop_handler); signal(SIGTERM, stop_handler);
    set_priority(30); /* vision=20, supervisor=10; highest application base priority */
    server = ipc_server_open(CUE_CHANNEL_NAME);
    if (!server) { perror("ipc_server_open"); return 1; }
    start = last_heartbeat = monotonic_ns();
    printf("DRIVER ready channel=%s\n", CUE_CHANNEL_NAME);
    while (!stopping) {
        cue_msg_t msg;
        int pulse, rcvid, timeout = 10;
        uint64_t now = monotonic_ns(), deadline;
        if (run_for > 0 && (double)(now - start) / 1e9 >= run_for) break;
        buzzer_tick(now);
        if (!faulted && now - last_heartbeat >= UINT64_C(300000000)) {
            faulted = 1;
            printf("WATCHDOG fault gap_ms=%.3f t_ns=%llu\n",
                   (double)(now - last_heartbeat) / 1e6, (unsigned long long)now);
            buzzer_pattern(CUE_FAULT);
        }
        deadline = faulted ? now + 10000000 : last_heartbeat + 300000000;
        if (pattern && transition < deadline) deadline = transition;
        if (deadline <= now) timeout = 0;
        else if (deadline - now < 10000000)
            timeout = (int)((deadline - now + 999999) / 1000000);
        rcvid = ipc_receive(server, &msg, &pulse, timeout);
#ifdef __QNX__
        /* name_attach's channel inherits sender priority on receive, even
         * from a lower-priority sender. Restore the driver's priority before
         * handling a cue, watchdog, or buzzer transition. */
        set_priority(30);
#endif
        now = monotonic_ns();
        if (rcvid == 0 && pulse) {
            last_heartbeat = now;
            if (faulted) printf("WATCHDOG recovered t_ns=%llu\n", (unsigned long long)now);
            faulted = 0;
        } else if (rcvid > 0) {
            int valid = msg.type == CUE_MSG_TYPE && msg.dir <= CUE_FAULT &&
                        msg.t_capture <= now;
            if (valid) {
                buzzer_pattern(msg.dir);
                now = monotonic_ns();
                printf("CUE frame=%u dir=%s latency_ms=%.3f t_ns=%llu\n",
                       (unsigned)msg.frame_id, cue_name(msg.dir),
                       (double)(now - msg.t_capture) / 1e6, (unsigned long long)now);
            }
            if (ipc_reply(server, rcvid, valid ? 0 : EINVAL) < 0)
                perror("ipc_reply");
        } else if (rcvid == -1 && errno != EINTR) perror("ipc_receive");
    }
    if (output_on) buzzer_output(0);
    ipc_server_close(server);
    printf("DRIVER stopped\n");
    return 0;
}
