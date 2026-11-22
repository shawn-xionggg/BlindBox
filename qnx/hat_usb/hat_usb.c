/*
 * hat_usb: talks to the ESP32-S3 buzzer hat over USB on QNX, with no serial driver.
 *
 * The QNX Pi image has no USB-serial (CDC-ACM) driver, so this program is the driver:
 * it uses QNX's libusbdi to find the hat's USB-serial chip (WCH CH343, 1a86:55d3, the
 * ESP32 DevKitC "UART" port), sets 115200 8N1 with DTR/RTS low (so the ESP32 isn't reset),
 * and writes the same one-byte protocol the laptop uses (esp32_cue.py):
 *   L R D B I = cues   H = heartbeat   S = stop   F = fault   X = goodbye   P -> K = ping
 *
 * Usage (on the Pi):
 *   hat_usb I            send bytes (any of LRDBIHSFX), e.g. "hat_usb R" buzzes the left ear
 *   hat_usb test         front, left, right, all, one after another
 *   hat_usb bench [N]    silent: time N cue sends (heartbeats), prints min/avg/max
 *   hat_usb ping [N]     N round trips P -> K (replies don't come back yet without HAT_SETUP)
 *   hat_usb fault        2 s of heartbeats, then silence: the hat must sound the fault alarm
 *   hat_usb -            forward bytes from stdin (one cue letter per byte)
 *
 * Runs as a normal user process: if it crashes, the kernel and the USB stack keep running.
 */
#include <errno.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/usbdi.h>

#define HAT_VENDOR  0x1a86   /* WCH */
#define HAT_PRODUCT 0x55d3   /* CH343 "USB Single Serial" */

#define CDC_SET_LINE_CODING        0x20
#define CDC_SET_CONTROL_LINE_STATE 0x22

static struct usbd_device *dev;
static struct usbd_pipe *ctrl_pipe, *out_pipe, *in_pipe;
static int comm_iface = 0;
static volatile int ready;
static volatile int attach_err;
static struct usbd_urb *urb;
static uint8_t *iobuf;

static void sleep_ms(int ms) { usleep(ms * 1000); }

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/*
 * QNX hands a composite USB device to drivers one interface at a time (one "instance" each).
 * The CH343 has a control interface (CDC class 0x02, no bulk endpoints) and a data interface
 * (class 0x0A, bulk IN + OUT). Attach to every instance and open the bulk pipes on whichever
 * interface the instance is for.
 */
static void insertion(struct usbd_connection *conn, usbd_device_instance_t *inst) {
    struct usbd_desc_node *dnode, *cnode, *inode, *enode;
    usbd_descriptors_t *d, *ifd, *ep;
    struct usbd_device *dv;
    int i, j, rc;

    if ((rc = usbd_attach(conn, inst, 0, &dv)) != EOK) {
        if (getenv("HAT_DEBUG")) fprintf(stderr, "attach iface %u: %s\n", (unsigned)inst->iface, strerror(rc));
        return;
    }
    if (!dev) dev = dv;

    d = usbd_parse_descriptors(dv, NULL, USB_DESC_DEVICE, 0, &dnode);
    if (d && !ctrl_pipe) usbd_open_pipe(dv, d, &ctrl_pipe);
    if (!usbd_parse_descriptors(dv, dnode, USB_DESC_CONFIGURATION, 0, &cnode)) return;

    for (i = 0; (ifd = usbd_parse_descriptors(dv, cnode, USB_DESC_INTERFACE, i, &inode)) != NULL; i++) {
        if (ifd->interface.bInterfaceClass == 0x02) comm_iface = ifd->interface.bInterfaceNumber;
        if (inst->iface != USBD_CONNECT_WILDCARD && ifd->interface.bInterfaceNumber != inst->iface) continue;
        if (getenv("HAT_DEBUG"))
            fprintf(stderr, "instance iface %u: class 0x%02x, %d endpoints\n", (unsigned)inst->iface,
                    ifd->interface.bInterfaceClass, ifd->interface.bNumEndpoints);
        for (j = 0; (ep = usbd_parse_descriptors(dv, inode, USB_DESC_ENDPOINT, j, &enode)) != NULL; j++) {
            int prc = EOK;
            if (USB_ENDPOINT_TYPE(ep->endpoint.bmAttributes) != USB_ATTRIB_BULK) continue;
            if (USB_ENDPOINT_DIR_IN(ep->endpoint.bEndpointAddress)) {
                if (!in_pipe) prc = usbd_open_pipe(dv, ep, &in_pipe);
            } else {
                if (!out_pipe) prc = usbd_open_pipe(dv, ep, &out_pipe);
            }
            if (getenv("HAT_DEBUG"))
                fprintf(stderr, "  bulk endpoint 0x%02x, max packet %d, open: %s\n", ep->endpoint.bEndpointAddress,
                        ep->endpoint.wMaxPacketSize, strerror(prc));
        }
    }
    if (out_pipe && ctrl_pipe) ready = 1;
}

static void removal(struct usbd_connection *conn, usbd_device_instance_t *inst) {
    (void)conn; (void)inst;
    fprintf(stderr, "hat_usb: hat unplugged\n");
    exit(2);
}

/*
 * usbd_io() only QUEUES a transfer and returns at once (500 sends "took" 0.016 ms each).
 * Reusing the URB/buffer while a transfer is still in flight corrupted io-usb-otg and crashed
 * the USB stack. So every transfer gets a completion callback and we wait for it here.
 */
static sem_t io_done;
static int urb_result(int rc, uint32_t *len);

static void io_complete(struct usbd_urb *u, struct usbd_pipe *p, void *h) {
    (void)u; (void)p; (void)h;
    sem_post(&io_done);
}

static int do_io(struct usbd_pipe *pipe, uint32_t timeout_ms, uint32_t *len) {
    struct timespec until;
    int rc;
    while (sem_trywait(&io_done) == 0) {}   /* clear any stale completion */
    if ((rc = usbd_io(urb, pipe, io_complete, NULL, timeout_ms)) != EOK) return rc;
    clock_gettime(CLOCK_REALTIME, &until);
    until.tv_sec += (timeout_ms + 200) / 1000;
    until.tv_nsec += ((timeout_ms + 200) % 1000) * 1000000L;
    if (until.tv_nsec >= 1000000000L) { until.tv_sec++; until.tv_nsec -= 1000000000L; }
    if (sem_timedwait(&io_done, &until) != 0) {
        usbd_abort_pipe(pipe);           /* the stack calls io_complete for the aborted URB */
        clock_gettime(CLOCK_REALTIME, &until);
        until.tv_sec += 1;
        sem_timedwait(&io_done, &until);
        return ETIMEDOUT;
    }
    return urb_result(EOK, len);
}

static int urb_result(int rc, uint32_t *len) {
    uint32_t status = 0, n = 0;
    if (rc != EOK) return rc;
    usbd_urb_status(urb, &status, &n);
    if (len) *len = n;
    if ((status & USBD_STATUS_CMP_ERR) || (status & USBD_STATUS_TIMEOUT) || (status & USBD_STATUS_ABORTED))
        return (status & USBD_STATUS_TIMEOUT) ? ETIMEDOUT : EIO;
    return EOK;
}

static int cdc_request(uint8_t request, uint16_t value, const void *data, uint32_t len) {
    if (len) memcpy(iobuf, data, len);
    /* always hand the stack a real DMA buffer, even for a request with no data */
    usbd_setup_vendor(urb, URB_DIR_OUT, request,
                      USB_TYPE_CLASS | USB_RECIPIENT_INTERFACE, value, (uint16_t)comm_iface,
                      iobuf, len);
    return do_io(ctrl_pipe, 1000, NULL);
}

static int send_bytes(const char *s, uint32_t n) {
    memcpy(iobuf, s, n);
    usbd_setup_bulk(urb, URB_DIR_OUT, iobuf, n);
    return do_io(out_pipe, 500, NULL);
}

/*
 * Reads until byte `want` shows up or timeout_ms passes; returns 1 if seen.
 * Never retries a failed read: a tight retry loop exhausts io-usb-otg's request pool
 * ("obj pool surb limit reached") and takes the whole USB stack down. A timed-out read
 * is cancelled with usbd_abort_pipe so it doesn't stay queued.
 */
static int wait_for(char want, double timeout_ms) {
    double end = now_ms() + timeout_ms;
    int reads = 0;
    if (!in_pipe) return 0;
    while (now_ms() < end && reads++ < 64) {
        uint32_t n = 0;
        uint32_t left = (uint32_t)(end - now_ms()) + 1;
        int rc;
        usbd_setup_bulk(urb, URB_DIR_IN | URB_SHORT_XFER_OK, iobuf, 512);
        rc = do_io(in_pipe, left, &n);
        if (rc == ETIMEDOUT) return 0;
        if (rc != EOK) {
            if (getenv("HAT_DEBUG")) fprintf(stderr, "read failed: %s\n", strerror(rc));
            return 0;
        }
        if (n && memchr(iobuf, want, n)) return 1;
        if (!n) sleep_ms(1);
    }
    return 0;
}

static int open_hat(int argc, char **argv) {
    static usbd_device_ident_t ident = { HAT_VENDOR, HAT_PRODUCT,
        USBD_CONNECT_WILDCARD, USBD_CONNECT_WILDCARD, USBD_CONNECT_WILDCARD };
    static usbd_funcs_t funcs = { _USBDI_NFUNCS, insertion, removal, NULL };
    usbd_connect_parm_t parm;
    struct usbd_connection *conn;
    uint8_t line[7] = { 0x00, 0xC2, 0x01, 0x00, 0, 0, 8 };   /* 115200 baud, 1 stop bit, no parity, 8 bits */
    double end;
    int rc;

    memset(&parm, 0, sizeof parm);
    parm.path = "/dev/usb/io-usb-otg";   /* QNX 8 USB stack (same path the camera config uses) */
    parm.vusb = USB_VERSION;
    parm.vusbd = USBD_VERSION;
    parm.argc = argc;
    parm.argv = argv;
    parm.ident = &ident;
    parm.funcs = &funcs;
    parm.connect_wait = 3;   /* seconds to wait for the USB stack, not the default 60 */
    if ((rc = usbd_connect(&parm, &conn)) != EOK) {
        fprintf(stderr, "hat_usb: can't reach the QNX USB stack: %s\n", strerror(rc));
        return -1;
    }
    for (end = now_ms() + 3000; !ready && now_ms() < end;) sleep_ms(10);
    if (ready != 1) {
        fprintf(stderr, "hat_usb: hat (USB %04x:%04x) %s\n", HAT_VENDOR, HAT_PRODUCT,
                ready ? strerror(attach_err) : "not found - is the ESP32's UART port plugged in?");
        return -1;
    }
    sem_init(&io_done, 0, 0);
    urb = usbd_alloc_urb(NULL);
    iobuf = usbd_alloc(512);
    if (!urb || !iobuf) { fprintf(stderr, "hat_usb: out of USB memory\n"); return -1; }
    /*
     * OFF by default: on QNX 8.0 / Pi 5, sending CDC SET_LINE_CODING to the CH343 crashes
     * io-usb-otg (the whole USB stack, camera included). Without it the CH343's default line
     * settings already match the ESP32 (verified: all four buzzers sound). HAT_SETUP=1 re-enables.
     */
    if (getenv("HAT_SETUP")) {
        if ((rc = cdc_request(CDC_SET_LINE_CODING, 0, line, sizeof line)) != EOK)
            fprintf(stderr, "hat_usb: warning: set 115200 baud failed: %s\n", strerror(rc));
        if ((rc = cdc_request(CDC_SET_CONTROL_LINE_STATE, 0, NULL, 0)) != EOK)   /* DTR=0 RTS=0: don't reset the ESP32 */
            fprintf(stderr, "hat_usb: warning: set DTR/RTS failed: %s\n", strerror(rc));
    }
    if (getenv("HAT_DEBUG")) fprintf(stderr, "setup done (comm iface %d)\n", comm_iface);
    return 0;
}

static int cue(char c) {
    int rc = send_bytes(&c, 1);
    if (rc != EOK) fprintf(stderr, "hat_usb: send '%c' failed: %s\n", c, strerror(rc));
    return rc;
}

/* ---- library API (vision links this file with -DHAT_LIB, see hat.h) ---- */
int hat_open(void) {
    static char name[] = "hat";
    static char *argv0[] = { name, NULL };
    return open_hat(1, argv0);
}

int hat_send(char c) { return send_bytes(&c, 1); }

#ifndef HAT_LIB
int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "test";
    setvbuf(stdout, NULL, _IOLBF, 0);   /* print each line right away, even into a log file */
    if (open_hat(argc, argv) != 0) return 1;

    if (strcmp(mode, "ping") == 0) {
        int n = argc > 2 ? atoi(argv[2]) : 20, ok = 0, i;
        double lo = 1e9, hi = 0, sum = 0;
        for (i = 0; i < n; i++) {
            double t0 = now_ms(), dt;
            if (cue('P') != EOK) break;
            if (!wait_for('K', 500)) { printf("ping %d: no reply\n", i + 1); continue; }
            dt = now_ms() - t0;
            ok++; sum += dt; if (dt < lo) lo = dt; if (dt > hi) hi = dt;
            sleep_ms(20);
        }
        if (ok) printf("ping: %d/%d replies, round trip min %.2f / avg %.2f / max %.2f ms\n", ok, n, lo, sum / ok, hi);
        else printf("ping: no replies (writes may still work: try 'hat_usb test')\n");
        return ok ? 0 : 1;
    }
    if (strcmp(mode, "test") == 0) {
        static const struct { char c; const char *what; } seq[] = {
            { 'I', "front buzzer (incoming / duck)" },
            { 'R', "LEFT ear  (punch from your left -> dodge right)" },
            { 'L', "RIGHT ear (punch from your right -> dodge left)" },
            { 'B', "all three (uppercut -> step back)" },
        };
        unsigned i;
        for (i = 0; i < sizeof seq / sizeof seq[0]; i++) {
            double t0 = now_ms();
            if (cue(seq[i].c) != EOK) return 1;
            printf("'%c' -> %-50s sent in %.2f ms\n", seq[i].c, seq[i].what, now_ms() - t0);
            sleep_ms(1200);
        }
        cue('S');
        return 0;
    }
    if (strcmp(mode, "bench") == 0) {   /* silent: N heartbeats, time each USB send */
        int n = argc > 2 ? atoi(argv[2]) : 200, i;
        double lo = 1e9, hi = 0, sum = 0;
        for (i = 0; i < n; i++) {
            double t0 = now_ms(), dt;
            if (cue('H') != EOK) return 1;
            dt = now_ms() - t0; sum += dt; if (dt < lo) lo = dt; if (dt > hi) hi = dt;
            sleep_ms(5);
        }
        cue('X');
        printf("bench: %d sends, min %.3f / avg %.3f / max %.3f ms per cue byte\n", n, lo, sum / n, hi);
        return 0;
    }
    if (strcmp(mode, "fault") == 0) {
        int i;
        printf("heartbeats for 2 s...\n"); fflush(stdout);
        for (i = 0; i < 40; i++) { if (cue('H') != EOK) return 1; sleep_ms(50); }
        printf("silence now: the hat should sound the FAULT alarm in ~0.6 s\n"); fflush(stdout);
        sleep_ms(2500);
        cue('X');
        printf("sent goodbye (X): alarm off\n");
        return 0;
    }
    if (strcmp(mode, "-") == 0) {
        int c;
        while ((c = getchar()) != EOF)
            if (c > ' ' && cue((char)c) != EOK) return 1;
        return 0;
    }
    /* otherwise: send the given letters */
    for (const char *p = mode; *p; p++) {
        if (!strchr("LRDBIHSFXP", *p)) { fprintf(stderr, "hat_usb: unknown cue '%c'\n", *p); return 1; }
        if (cue(*p) != EOK) return 1;
    }
    return 0;
}
#endif /* HAT_LIB */
