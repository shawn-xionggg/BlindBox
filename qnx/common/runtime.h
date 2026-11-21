#ifndef DODGE_RUNTIME_H
#define DODGE_RUNTIME_H
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <errno.h>
#endif
#ifdef __QNX__
#include <pthread.h>
#include <sched.h>
#include <string.h>
#endif

static inline uint64_t monotonic_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}
static inline void sleep_ms(unsigned ms)
{
#ifdef _WIN32
    Sleep(ms);
#else
    struct timespec ts = {(time_t)(ms / 1000), (long)(ms % 1000) * 1000000L};
    while (nanosleep(&ts, &ts) < 0 && errno == EINTR) {}
#endif
}
static inline void set_priority(int priority)
{
#ifdef __QNX__
    struct sched_param param;
    int err;
    memset(&param, 0, sizeof(param));
    param.sched_priority = priority;
    err = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    if (err) fprintf(stderr, "SCHED_FIFO priority=%d failed: %s\n", priority, strerror(err));
#else
    (void)priority; /* Windows/socket tests do not verify real-time scheduling. */
#endif
}
#endif
