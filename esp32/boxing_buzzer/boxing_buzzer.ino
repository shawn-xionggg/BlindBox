/*
  Blindfolded Dodge - ESP32 with 3 buzzers on the hat: FRONT, LEFT ear, RIGHT ear.

  The laptop (dodge_live.py --esp32 ...) sends ONE BYTE per cue, the moment the
  detector decides. The ESP32 starts the right buzzer(s) immediately (nothing
  blocks or waits) and stops them on its own after the cue's length.

  Which buzzer plays (default MOVE_TOWARD_BUZZ = true: "move toward the buzz"):
    'L' DODGE LEFT   -> LEFT ear            'R' DODGE RIGHT -> RIGHT ear
    'D' DUCK         -> FRONT               'B' STEP BACK   -> ALL THREE
    'I' INCOMING (one-beep mode) -> FRONT
  With MOVE_TOWARD_BUZZ = false the ears swap: the buzz is on the side the punch
  comes FROM, so you move away from it. Try both; keep the faster one.

  Other bytes:
    'H' heartbeat (every video frame)   'P' ping -> replies 'K' (latency test)
    'S' stop sound   'F' fault sound once   'X' goodbye (clean exit -> no alarm)

  Two ways in, both always on:
    USB serial at 115200 baud (lowest delay)
    WiFi UDP port 4210 on the ESP32's own hotspot "dodge-hat" (no venue WiFi needed)

  Safety net: once heartbeats have started, if NOTHING arrives for 600 ms (laptop
  crashed, cable pulled, WiFi dropped), all buzzers chirp once a second until
  bytes come back.

  Wiring (classic ESP32 dev board; see esp32/README.md):
    FRONT buzzer +  -> GPIO 25      LEFT ear +  -> GPIO 26      RIGHT ear +  -> GPIO 27
    all three buzzers -  -> GND
  Passive buzzers (need a tone): ACTIVE_BUZZER = false.  Active (beep on plain power): true.
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#if defined(__has_include)
#if __has_include(<esp_arduino_version.h>)
#include <esp_arduino_version.h>
#endif
#endif
#ifndef ESP_ARDUINO_VERSION_MAJOR
#define ESP_ARDUINO_VERSION_MAJOR 2
#endif

// ------------------------------------------------------------------ settings
const int      PIN_FRONT     = 25;
const int      PIN_LEFT      = 26;
const int      PIN_RIGHT     = 27;
const bool     ACTIVE_BUZZER = false;
const bool     MOVE_TOWARD_BUZZ = true;
const bool     USE_WIFI      = true;
const char*    AP_SSID       = "dodge-hat";
const char*    AP_PASS       = "dodgehat123";   // must be 8+ characters
const uint16_t UDP_PORT      = 4210;
const uint32_t HEARTBEAT_TIMEOUT_MS = 600;
// ---------------------------------------------------------------------------

enum { BZ_FRONT = 1, BZ_LEFT = 2, BZ_RIGHT = 4, BZ_ALL = 7 };
const int PINS[3]     = {PIN_FRONT, PIN_LEFT, PIN_RIGHT};
const int CHANNELS[3] = {0, 2, 4};   // LEDC channels on separate timers (core 2.x)

struct Cue { char code; uint8_t mask; uint16_t freq; uint16_t ms; };
// Same pitches/lengths as the laptop beeps in dodge_live.py.
Cue CUES[] = {
  {'L', BZ_LEFT,  600, 120},   // DODGE LEFT
  {'R', BZ_RIGHT, 1200, 120},  // DODGE RIGHT
  {'D', BZ_FRONT, 900, 250},   // DUCK
  {'B', BZ_ALL,   400, 350},   // STEP BACK
  {'I', BZ_FRONT, 1000, 150},  // INCOMING
};

WiFiUDP udp;
uint32_t toneEndMs = 0;        // 0 = silent
uint32_t lastRxMs = 0;
bool armed = false;            // true once heartbeats have started
uint32_t lastFaultChirpMs = 0;

// --- low-level: one PWM output per buzzer (all three can sound at once) ---
void outInit(int i) {
  pinMode(PINS[i], OUTPUT);
  digitalWrite(PINS[i], LOW);
  if (ACTIVE_BUZZER) return;
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(PINS[i], 1000, 8);
  ledcWrite(PINS[i], 0);
#else
  ledcSetup(CHANNELS[i], 1000, 8);
  ledcAttachPin(PINS[i], CHANNELS[i]);
  ledcWrite(CHANNELS[i], 0);
#endif
}

void outOn(int i, uint16_t freq) {
  if (ACTIVE_BUZZER) { digitalWrite(PINS[i], HIGH); return; }
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWriteTone(PINS[i], freq);
#else
  ledcWriteTone(CHANNELS[i], freq);
#endif
}

void outOff(int i) {
  if (ACTIVE_BUZZER) { digitalWrite(PINS[i], LOW); return; }
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWriteTone(PINS[i], 0);     // frequency 0 = silent
#else
  ledcWriteTone(CHANNELS[i], 0);
#endif
}

// --- cues ---
void soundOff() {
  for (int i = 0; i < 3; i++) outOff(i);
  toneEndMs = 0;
}

void play(uint8_t mask, uint16_t freq, uint16_t ms) {
  soundOff();                              // a new cue cuts off the old one
  for (int i = 0; i < 3; i++)
    if (mask & (1 << i)) outOn(i, freq);
  toneEndMs = millis() + ms;
  if (toneEndMs == 0) toneEndMs = 1;
}

void reply(char c, bool fromUdp) {
  if (fromUdp) {
    udp.beginPacket(udp.remoteIP(), udp.remotePort());
    udp.write((uint8_t)c);
    udp.endPacket();
  } else {
    Serial.write(c);
  }
}

void handleByte(char c, bool fromUdp) {
  lastRxMs = millis();
  switch (c) {
    case 'H': armed = true; return;
    case 'P': reply('K', fromUdp); return;
    case 'S': soundOff(); return;
    case 'X': armed = false; soundOff(); return;
    case 'F': play(BZ_ALL, 300, 400); return;
  }
  for (const Cue& cue : CUES) {
    if (cue.code == c) { play(cue.mask, cue.freq, cue.ms); return; }
  }
  // anything else (newlines etc.) is ignored
}

void setup() {
  if (!MOVE_TOWARD_BUZZ) {                 // buzz the side the punch comes FROM instead
    for (Cue& cue : CUES) {
      if (cue.code == 'L') cue.mask = BZ_RIGHT;
      if (cue.code == 'R') cue.mask = BZ_LEFT;
    }
  }
  for (int i = 0; i < 3; i++) outInit(i);
  Serial.begin(115200);
  if (USE_WIFI) {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASS);
    WiFi.setSleep(false);                  // no WiFi power-save: it adds delay
    udp.begin(UDP_PORT);
    Serial.printf("\nWiFi hotspot '%s' up, IP %s, UDP port %u\n",
                  AP_SSID, WiFi.softAPIP().toString().c_str(), UDP_PORT);
  }
  Serial.println("boxing_buzzer ready (3 buzzers)");
  // power-on check: front, left, right, one after another (so you can hear each is wired)
  const uint8_t order[3] = {BZ_FRONT, BZ_LEFT, BZ_RIGHT};
  for (int k = 0; k < 3; k++) { play(order[k], 1500, 80); delay(150); soundOff(); }
}

void loop() {
  // 1) USB serial
  while (Serial.available() > 0) handleByte((char)Serial.read(), false);

  // 2) WiFi UDP (a packet may hold several bytes)
  if (USE_WIFI) {
    int n = udp.parsePacket();
    while (n > 0) {
      uint8_t buf[16];
      int got = udp.read(buf, sizeof(buf));
      for (int i = 0; i < got; i++) handleByte((char)buf[i], true);
      n = udp.parsePacket();
    }
  }

  uint32_t now = millis();

  // 3) end the current sound on time
  if (toneEndMs != 0 && (int32_t)(now - toneEndMs) >= 0) soundOff();

  // 4) heartbeat watchdog: laptop gone -> all buzzers chirp once a second
  if (armed && now - lastRxMs > HEARTBEAT_TIMEOUT_MS) {
    if (now - lastFaultChirpMs >= 1000) {
      play(BZ_ALL, 300, 80);
      lastFaultChirpMs = now;
    }
  }
}
