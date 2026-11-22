/* Buzzer hat over USB on QNX (implementation: hat_usb.c built with -DHAT_LIB). */
#ifndef HAT_H
#define HAT_H
/* Finds the hat (CH343 1a86:55d3) through io-usb-otg. 0 = ok, -1 = not found (message printed). */
int hat_open(void);
/* Sends one protocol byte and waits until the USB transfer completes. 0 = ok, else an errno. */
int hat_send(char c);
#endif
