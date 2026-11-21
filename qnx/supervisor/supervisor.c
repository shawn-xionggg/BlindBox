#include "runtime.h"
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

/* Custom supervisor for the demo. QNX High Availability Manager (HAM) is
 * the production alternative for monitoring and restart policies. */
typedef struct {
#ifdef _WIN32
    HANDLE handle;
    DWORD pid;
#else
    pid_t pid;
#endif
    int live;
} child_t;
static volatile sig_atomic_t stopping;
static void stop_handler(int sig) { (void)sig; stopping = 1; }

#ifdef _WIN32
/* Quote argv using Windows CRT backslash/quote rules, without a shell. */
static int quote_arg(char *out, size_t capacity, size_t *used, const char *arg)
{
    const char *p = arg;
    if (*used + 3 >= capacity) return -1;
    out[(*used)++] = '"';
    for (;;) {
        size_t slashes = 0, copies;
        while (*p == '\\') { ++slashes; ++p; }
        copies = (*p == '"' || !*p) ? slashes * 2 : slashes;
        if (*used + copies + 4 >= capacity) return -1;
        while (copies--) out[(*used)++] = '\\';
        if (!*p) break;
        if (*p == '"') out[(*used)++] = '\\';
        out[(*used)++] = *p++;
    }
    out[(*used)++] = '"'; out[(*used)++] = ' '; out[*used] = '\0';
    return 0;
}
#endif
static int start_child(child_t *child, char *const args[], const char *vision_cmd)
{
#ifdef _WIN32
    STARTUPINFOA startup;
    PROCESS_INFORMATION process;
    char command[8192] = {0};
    size_t used = 0;
    int i;
    memset(&startup, 0, sizeof(startup)); startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    memset(&process, 0, sizeof(process));
    if (vision_cmd) {
        if (strlen(vision_cmd) >= sizeof(command)) return -1;
        strcpy(command, vision_cmd);
    } else {
        for (i = 0; args[i]; ++i)
            if (quote_arg(command, sizeof(command), &used, args[i]) < 0) return -1;
    }
    /* Inherit redirected stdout/stderr so tests see the whole process tree. */
    if (!CreateProcessA(vision_cmd ? NULL : args[0], command, NULL, NULL, TRUE, CREATE_NO_WINDOW,
                        NULL, NULL, &startup, &process)) {
        fprintf(stderr, "CreateProcess failed error=%lu\n", (unsigned long)GetLastError());
        return -1;
    }
    CloseHandle(process.hThread);
    child->handle = process.hProcess; child->pid = process.dwProcessId;
#else
    int err;
    if (vision_cmd) {
        /* exec keeps the executable as our direct child: wait/kill own it.
         * Accept shell quoting for a single executable plus its arguments. */
        char command[8192];
        char *shell_args[] = {"/bin/sh", "-c", command, NULL};
        if (strlen(vision_cmd) + 6 > sizeof(command)) return -1;
        snprintf(command, sizeof(command), "exec %s", vision_cmd);
        err = posix_spawn(&child->pid, "/bin/sh", NULL, NULL, shell_args, environ);
    } else err = posix_spawn(&child->pid, args[0], NULL, NULL, args, environ);
    if (err) { errno = err; perror("posix_spawn"); return -1; }
#endif
    child->live = 1;
    return 0;
}
/* 0 running, 1 exited (reaped), -1 monitor failure. */
static int poll_child(child_t *child, int *status)
{
    if (!child->live) return 0;
#ifdef _WIN32
    DWORD code, wait = WaitForSingleObject(child->handle, 0);
    if (wait == WAIT_TIMEOUT) return 0;
    if (wait != WAIT_OBJECT_0 || !GetExitCodeProcess(child->handle, &code)) return -1;
    *status = (int)code; CloseHandle(child->handle);
#else
    int raw;
    pid_t result = waitpid(child->pid, &raw, WNOHANG);
    if (result == 0 || (result < 0 && errno == EINTR)) return 0;
    if (result < 0) return -1;
    *status = WIFEXITED(raw) ? WEXITSTATUS(raw) : 128 + WTERMSIG(raw);
#endif
    child->live = 0; return 1;
}
static void stop_child(child_t *child)
{
    if (!child->live) return;
#ifdef _WIN32
    TerminateProcess(child->handle, 0);
    WaitForSingleObject(child->handle, INFINITE);
    CloseHandle(child->handle);
#else
    unsigned i;
    int status;
    kill(child->pid, SIGTERM);
    for (i = 0; i < 100; ++i) {
        if (poll_child(child, &status) == 1) return;
        sleep_ms(10);
    }
    kill(child->pid, SIGKILL);
    while (waitpid(child->pid, NULL, 0) < 0 && errno == EINTR) {}
#endif
    child->live = 0;
}
static int parse_seconds(const char *s, double *out)
{
    char *end;
    *out = strtod(s, &end);
    return end != s && !*end && *out >= 0 && *out <= 86400;
}
int main(int argc, char **argv)
{
    const char *csv = NULL;
    const char *vision_cmd = NULL;
    double duration = 5, crash_at = 0;
    child_t driver = {0}, vision = {0};
    char driver_path[4096], vision_path[4096], crash_text[40], duration_text[40];
    char *driver_args[] = {driver_path, NULL};
    char *vision_args[] = {vision_path, "--csv", NULL, "--run-for", duration_text,
                           "--crash-at", crash_text, NULL};
    uint64_t start, restart_due = 0, died_at = 0;
    unsigned restarts = 0;
    int i, status, result = 0;
    const char *last_sep = NULL, *p;
    size_t prefix;
    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else if (!strcmp(argv[i], "--vision-cmd") && i + 1 < argc) vision_cmd = argv[++i];
        else if (!strcmp(argv[i], "--duration") && i + 1 < argc) {
            if (!parse_seconds(argv[++i], &duration) || duration == 0) return 2;
        } else if (!strcmp(argv[i], "--crash-at") && i + 1 < argc) {
            if (!parse_seconds(argv[++i], &crash_at)) return 2;
        } else {
            fprintf(stderr, "usage: supervisor (--csv events.csv | --vision-cmd command) [--duration s] [--crash-at s]\n");
            return 2;
        }
    }
    if (vision_cmd) {
        if (!*vision_cmd || csv || crash_at != 0) {
            fprintf(stderr, "--vision-cmd must be nonempty and cannot combine with --csv/--crash-at\n");
            return 2;
        }
    } else {
        FILE *fp;
        if (!csv) { fprintf(stderr, "--csv or --vision-cmd is required\n"); return 2; }
        fp = fopen(csv, "r"); if (!fp) { perror(csv); return 2; } fclose(fp);
    }
    for (p = argv[0]; *p; ++p) if (*p == '/' || *p == '\\') last_sep = p;
    prefix = last_sep ? (size_t)(last_sep - argv[0] + 1) : 0;
    if (prefix > sizeof(driver_path) - 32) return 2;
#ifdef _WIN32
#define EXE_SUFFIX ".exe"
#else
#define EXE_SUFFIX ""
#endif
    snprintf(driver_path, sizeof(driver_path), "%.*scue_driver%s", (int)prefix, argv[0], EXE_SUFFIX);
    snprintf(vision_path, sizeof(vision_path), "%.*svision_stub%s", (int)prefix, argv[0], EXE_SUFFIX);
    /* When launched without a directory, use cwd (documented in README). */
#ifndef _WIN32
    if (!prefix) {
        strcpy(driver_path, "./cue_driver"); strcpy(vision_path, "./vision_stub");
    }
#endif
    snprintf(crash_text, sizeof(crash_text), "%.6f", crash_at);
    snprintf(duration_text, sizeof(duration_text), "%.6f", duration + 1);
    vision_args[2] = (char *)csv;
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGINT, stop_handler); signal(SIGTERM, stop_handler);
    set_priority(10);
    if (start_child(&driver, driver_args, NULL) < 0) return 1;
    printf("SUPERVISOR driver_started pid=%lu\n", (unsigned long)driver.pid);
    sleep_ms(75); /* Stub also retries connection for up to two seconds. */
    if (start_child(&vision, vision_args, vision_cmd) < 0) { stop_child(&driver); return 1; }
    printf("SUPERVISOR vision_started pid=%lu\n", (unsigned long)vision.pid);
    start = monotonic_ns();
    while (!stopping && (double)(monotonic_ns() - start) / 1e9 < duration) {
        uint64_t now = monotonic_ns();
        int polled = poll_child(&driver, &status);
        if (polled != 0) {
            fprintf(stderr, "SUPERVISOR cue_driver unavailable\n"); result = 1; break;
        }
        polled = poll_child(&vision, &status);
        if (polled < 0) { fprintf(stderr, "SUPERVISOR wait failed\n"); result = 1; break; }
        if (polled == 1) {
            died_at = now; restart_due = now + 650000000;
            printf("SUPERVISOR vision_exit status=%d t_ns=%llu\n", status, (unsigned long long)now);
        }
        if (!vision.live && restart_due && now >= restart_due) {
            strcpy(crash_text, "0"); /* Inject exactly one crash, not a crash loop. */
            if (start_child(&vision, vision_args, vision_cmd) < 0) { result = 1; break; }
            ++restarts; restart_due = 0;
            now = monotonic_ns();
            printf("SUPERVISOR restart=%u pid=%lu delay_ms=%.3f t_ns=%llu\n",
                   restarts, (unsigned long)vision.pid, (double)(now - died_at) / 1e6,
                   (unsigned long long)now);
        }
        sleep_ms(10);
    }
    stop_child(&vision); stop_child(&driver);
    printf("SUPERVISOR stopped restarts=%u\n", restarts);
    return result;
}
