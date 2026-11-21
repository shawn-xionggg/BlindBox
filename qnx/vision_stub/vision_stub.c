#include "ipc.h"
#include "runtime.h"
#include <ctype.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>

typedef struct { double time_s; unsigned dir; } event_t;
static volatile sig_atomic_t stopping;
static void stop_handler(int sig) { (void)sig; stopping = 1; }
static int direction(const char *s)
{
    unsigned i;
    if (strncmp(s, "CUE_", 4) == 0) s += 4;
    for (i = 0; i <= CUE_FAULT; ++i)
        if (strcmp(s, cue_name(i)) == 0) return (int)i;
    if (strlen(s) == 1 && s[0] >= '0' && s[0] <= '0' + CUE_FAULT)
        return s[0] - '0';
    return -1;
}
static int load_csv(const char *path, event_t **events, size_t *count)
{
    FILE *fp = fopen(path, "r");
    char line[256];
    unsigned line_no = 0;
    size_t capacity = 0;
    if (!fp) { perror(path); return -1; }
    *events = NULL; *count = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *p = line, *end, *tail;
        double when;
        int dir;
        ++line_no;
        if (!strchr(line, '\n') && !feof(fp)) goto invalid;
        while (isspace((unsigned char)*p)) ++p;
        if (!*p || *p == '#') continue;
        if (strncmp(p, "time_s,", 7) == 0 && *count == 0) continue;
        when = strtod(p, &end);
        if (end == p || !(when >= 0) || when > 86400) goto invalid;
        while (isspace((unsigned char)*end)) ++end;
        if (*end++ != ',') goto invalid;
        while (isspace((unsigned char)*end)) ++end;
        tail = end + strlen(end);
        while (tail > end && isspace((unsigned char)tail[-1])) --tail;
        *tail = '\0'; dir = direction(end);
        if (dir < 0 || (*count && when < (*events)[*count - 1].time_s)) goto invalid;
        if (*count == capacity) {
            event_t *next;
            capacity = capacity ? capacity * 2 : 16;
            next = realloc(*events, capacity * sizeof(**events));
            if (!next) { fclose(fp); free(*events); *events = NULL; return -1; }
            *events = next;
        }
        (*events)[*count].time_s = when; (*events)[*count].dir = (unsigned)dir;
        ++*count;
    }
    if (ferror(fp)) { fclose(fp); free(*events); *events = NULL; return -1; }
    fclose(fp); return 0;
invalid:
    fprintf(stderr, "%s:%u: expected ordered time_s,dir CSV\n", path, line_no);
    fclose(fp); free(*events); *events = NULL; return -1;
}
static int seconds(const char *text, double *value)
{
    char *end;
    *value = strtod(text, &end);
    return end != text && !*end && *value >= 0 && *value <= 86400;
}
int main(int argc, char **argv)
{
    const char *csv = NULL;
    double crash_at = 0, run_for = 10;
    event_t *events;
    size_t count, next = 0;
    ipc_conn_t *conn = NULL;
    uint64_t start, next_heartbeat;
    uint32_t frame = 0;
    int i, result = 0;
    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else if (!strcmp(argv[i], "--crash-at") && i + 1 < argc) {
            if (!seconds(argv[++i], &crash_at)) return 2;
        } else if (!strcmp(argv[i], "--run-for") && i + 1 < argc) {
            if (!seconds(argv[++i], &run_for) || run_for == 0) return 2;
        } else {
            fprintf(stderr, "usage: vision_stub --csv events.csv [--crash-at s] [--run-for s]\n");
            return 2;
        }
    }
    if (!csv || load_csv(csv, &events, &count) < 0) return 2;
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGINT, stop_handler); signal(SIGTERM, stop_handler);
    set_priority(20);
    for (i = 0; i < 100 && !stopping; ++i) {
        conn = ipc_client_open(CUE_CHANNEL_NAME);
        if (conn) break;
        sleep_ms(20);
    }
    if (!conn) { fprintf(stderr, "STUB cannot connect\n"); free(events); return 1; }
    start = next_heartbeat = monotonic_ns();
    printf("STUB started events=%u crash_at=%.3f t_ns=%llu\n",
           (unsigned)count, crash_at, (unsigned long long)start);
    while (!stopping) {
        uint64_t now = monotonic_ns();
        double elapsed = (double)(now - start) / 1e9;
        if (crash_at > 0 && elapsed >= crash_at) {
            printf("STUB simulated_crash t_ns=%llu\n", (unsigned long long)now);
            _Exit(77); /* No orderly IPC close; exercise peer-death path. */
        }
        if (elapsed >= run_for) break;
        if (now >= next_heartbeat) {
            if (ipc_send_heartbeat(conn) < 0) { result = 1; break; }
            ++frame;
            next_heartbeat = now + 50000000; /* ~20 frames/second */
        }
        while (next < count && elapsed >= events[next].time_s) {
            cue_msg_t msg;
            memset(&msg, 0, sizeof(msg));
            msg.type = CUE_MSG_TYPE; msg.dir = (uint8_t)events[next].dir;
            msg.score = 1.0f; msg.frame_id = frame;
            msg.t_capture = monotonic_ns(); /* Synthetic capture, NOT camera exposure. */
            if (ipc_send_cue(conn, &msg) != 0) {
                fprintf(stderr, "STUB cue send failed event=%u\n", (unsigned)next);
                result = 1; goto done;
            }
            printf("STUB ack event=%u dir=%s\n", (unsigned)next, cue_name(msg.dir));
            ++next;
        }
        sleep_ms(2);
    }
done:
    ipc_client_close(conn); free(events);
    printf("STUB stopped\n"); return result;
}
