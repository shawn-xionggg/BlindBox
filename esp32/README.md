# ESP32 hat buzzers (front + left ear + right ear)

The laptop detects the punch and sends **one byte** to an ESP32 on the hat. The ESP32 starts the right buzzer(s) **immediately**: each cue's sound is pre-set in the firmware, and nothing waits or blocks.

**The buzz is on the side the punch comes from, so you flinch away from it.** That's the default, `MOVE_TOWARD_BUZZ = false`.

| Punch | Laptop cue | Byte | Buzzer(s) | Sound |
|---|---|---|---|---|
| Hook from your **left** | DODGE RIGHT | `R` | **left ear** | 1200 Hz, 120 ms |
| Hook from your **right** | DODGE LEFT | `L` | **right ear** | 600 Hz, 120 ms |
| Straight (jab / cross) | DUCK | `D` | **front** | active buzzer, 250 ms |
| Uppercut | STEP BACK | `B` | **all three** | 350 ms |
| Any (`--universal`) | INCOMING | `I` | front | 150 ms |

Set `MOVE_TOWARD_BUZZ = true` to swap the ears ("move toward the buzz").

- **Safety alarm:** the laptop also sends a heartbeat byte every video frame. If those stop for 600 ms (laptop crashed, cable pulled, WiFi dropped), **all three buzzers chirp once a second.** Quitting the demo normally sends a "goodbye" byte, so no alarm.
- **Power-on check:** front, then left, then right each chirp once, so you can hear that every buzzer is wired.

## 1. Wiring

**ESP32-S3-DevKitC-1** (the "N8R8", "N16R8" etc. part of the name is only the flash/PSRAM size, and the wiring is the same for all of them):

| Buzzer | + leg (long leg, or pin marked S / I/O / +) | − leg (short leg, or GND / −) |
|---|---|---|
| FRONT | **GPIO 4** (header pin labeled `4`) | GND |
| LEFT ear | **GPIO 5** (`5`) | GND |
| RIGHT ear | **GPIO 6** (`6`) | GND |

```
ESP32-S3                    hat
4    ───────────────── (+) FRONT buzzer (−) ──┐
5    ───────────────── (+) LEFT-ear buzzer (−)─┤
6    ───────────────── (+) RIGHT-ear buzzer (−)┤
GND  ──────────────────────────────────────────┘  (all three − legs share GND)
```

Pins 4, 5 and 6 sit next to each other on the same header, near the top (by the 3V3 / RST pins).

**Pins to avoid on the S3:**
- **26–37:** taken by the flash and PSRAM chips.
- **19 and 20:** the USB port.
- **0, 3, 45, 46:** affect how it boots.
- **43 and 44:** the serial port to the laptop.
- **48** (or **38** on newer boards): the RGB LED.

On a *classic* ESP32 board (no "S3" in the name), GPIO 6 is wired to the flash chip, so **don't** use 4/5/6 there. Use **GPIO 25, 26 and 27** and change `PIN_FRONT`, `PIN_LEFT` and `PIN_RIGHT` at the top of the sketch.

- **Current:** small piezo buzzers are fine straight on the pins. *Magnetic* buzzers (black, usually round, with a coil inside) draw about 30 mA each. Three of those straight on the pins is too much, so drive each through a transistor:
  ```
  GPIO ──[1 kΩ]── base of NPN (2N2222 / S8050)
  buzzer + ── 3.3 V (or 5 V for a louder buzzer)      buzzer − ── collector
  emitter ── GND
  ```
- **Wires to the ears:** run a twisted pair (signal + GND) to each ear so long wires don't pick up noise. Tape them along the hat band.
- **Active or passive?**
  - Each buzzer has its own setting at the top of the sketch:
    - `LEFT_IS_ACTIVE = false` and `RIGHT_IS_ACTIVE = false`: **the ear buzzers are passive**, so they play the cue pitches.
    - `FRONT_IS_ACTIVE = true`: **the front buzzer is active**, meaning it beeps by itself when powered. It always has its own fixed pitch, so DUCK and INCOMING differ only in length.
  - Not sure about the front one? Hold it to 3.3 V and GND: if it beeps on its own, it's active (`true`). If it's silent or just clicks, it's passive (`false`).

- Power the ESP32 from the hat's power bank by USB, or from the laptop's USB cable in USB mode.

## 2. Flash the firmware (Arduino IDE)

1. Install the Arduino IDE 2 from https://www.arduino.cc/en/software.
2. Add ESP32 support: **Tools → Board → Boards Manager**, search **"esp32"**, install **"esp32 by Espressif Systems"**.
3. Open `esp32/boxing_buzzer/boxing_buzzer.ino`.
4. Check the settings at the top: `PIN_FRONT` / `PIN_LEFT` / `PIN_RIGHT`, `FRONT_IS_ACTIVE` / `LEFT_IS_ACTIVE` / `RIGHT_IS_ACTIVE`, `MOVE_TOWARD_BUZZ`, the WiFi name and password.
5. **Tools → Board → esp32 → "ESP32S3 Dev Module".** Leave the other Tools settings at their defaults.
   - The DevKitC-1 has **two USB-C ports**. Plug the cable into the one labeled **`UART`** (or `COM`), not the one labeled `USB`. That's the port the laptop talks to the buzzer through.
   - Then choose **Tools → Port**: the COM port that appears when you plug it in.
6. Click **Upload**. If it hangs on "Connecting…", hold the board's **BOOT** button until uploading starts.
7. Optional check: open **Tools → Serial Monitor** at 115200 baud and press the board's EN/RST button. You should see `boxing_buzzer ready (3 buzzers)` and hear front → left → right chirp. **Then close the Serial Monitor:** only one program can use the port at a time.

## 3. Connect the laptop

You can use either connection, or both:

| | USB cable | WiFi |
|---|---|---|
| Delay | lowest (about 1–2 ms) | a few ms, with occasional spikes |
| Needs | `python -m pip install pyserial` | laptop joined to the ESP32's hotspot **`dodge-hat`** (password `dodgehat123`) |
| Target | `COM5` (your port) | `udp:192.168.4.1` |

**Find the port (USB):**
```powershell
python esp32_cue.py --list
```

**Test it.** This pings the ESP32 20 times, prints the round-trip delay, then plays every cue:
```powershell
python esp32_cue.py COM5 --test
python esp32_cue.py udp:192.168.4.1 --test
```

**Run the demo with the hat buzzer:**
```powershell
python dodge_live.py --source 1 --esp32 COM5 --no-beep
python dodge_live.py --source 1 --esp32 udp:192.168.4.1 --no-beep
```

- `--no-beep` silences the laptop speaker, so the only cue is the hat. Leave it off to hear both.
- The ESP32 byte is sent **before** anything else when a punch is detected.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No reply from the ESP32` | Firmware flashed? Right COM port? Serial Monitor closed? On WiFi: is the laptop actually joined to `dodge-hat`? |
| `USB mode needs pyserial` | `python -m pip install pyserial`, or use WiFi |
| Board resets when the demo starts | Normal on some boards (USB control lines). The code waits 1.5 s for it to boot. |
| One buzzer only clicks | That buzzer is active: set its `..._IS_ACTIVE = true` and re-upload |
| Front buzzer silent or very faint | It's probably passive: set `FRONT_IS_ACTIVE = false` and re-upload |
| One buzzer silent at power-on | Check that buzzer's wire and GND, and that its pin matches `PIN_FRONT` / `PIN_LEFT` / `PIN_RIGHT` |
| Left and right feel backwards | Swap the two ear wires, or flip `MOVE_TOWARD_BUZZ` |
| Chirps every second | It lost the laptop: the demo crashed or stopped, or the cable/WiFi dropped |
| Too quiet | Use a transistor driver and a bigger buzzer, or put the buzzer closer to the ear |

**Not tested yet:** the firmware hasn't been run on a real ESP32. It was compile-checked with a normal C++ compiler against stand-ins for the ESP32 libraries (both 2.x and 3.x), which catches typos but not hardware behavior. The laptop side was tested against a fake ESP32 over UDP: every cue, heartbeat, ping and goodbye byte arrived correctly.
