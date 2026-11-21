# ESP32 hat buzzers (front + left ear + right ear)

The laptop detects the punch and sends **one byte** to an ESP32 on the hat. The ESP32 starts the right buzzer(s) **immediately**: each cue's sound is pre-set in the firmware, and nothing waits or blocks.

| Cue | Byte | Buzzer(s) | Sound (passive buzzers) |
|---|---|---|---|
| DODGE LEFT | `L` | **left ear** | 600 Hz, 120 ms |
| DODGE RIGHT | `R` | **right ear** | 1200 Hz, 120 ms |
| DUCK | `D` | **front** | 900 Hz, 250 ms |
| STEP BACK | `B` | **all three** | 400 Hz, 350 ms |
| INCOMING (`--universal`) | `I` | front | 1000 Hz, 150 ms |

**"Move toward the buzz"** is the default: a buzz in the left ear means dodge left. People usually react faster when the move is on the same side as the signal. If the wearer prefers "the buzz is where the punch comes from, move away", set `MOVE_TOWARD_BUZZ = false`, which swaps the ears. Try both blindfolded and keep the faster one.

- **Safety alarm:** the laptop also sends a heartbeat byte every video frame. If those stop for 600 ms (laptop crashed, cable pulled, WiFi dropped), **all three buzzers chirp once a second.** Quitting the demo normally sends a "goodbye" byte, so no alarm.
- **Power-on check:** front, then left, then right each chirp once, so you can hear that every buzzer is wired.

## 1. Wiring

Classic **ESP32 dev board** (ESP32-WROOM / "ESP32 Dev Module"):

| Buzzer | + leg (long leg, or pin marked S / I/O / +) | − leg (short leg, or GND / −) |
|---|---|---|
| FRONT | **GPIO 25** | GND |
| LEFT ear | **GPIO 26** | GND |
| RIGHT ear | **GPIO 27** | GND |

```
ESP32                       hat
GPIO 25  ───────────────── (+) FRONT buzzer (−) ──┐
GPIO 26  ───────────────── (+) LEFT-ear buzzer (−)─┤
GPIO 27  ───────────────── (+) RIGHT-ear buzzer (−)┤
GND      ──────────────────────────────────────────┘  (all three − legs share GND)
```

- **ESP32-C3 or ESP32-S3** boards don't have GPIO 25–27. Use **GPIO 4, 5 and 6** instead, and change `PIN_FRONT`, `PIN_LEFT` and `PIN_RIGHT` at the top of the sketch.
- Avoid pins 0, 2, 12 and 15. They affect how the board boots.
- **Current:** small piezo buzzers are fine straight on the pins. *Magnetic* buzzers (black, usually round, with a coil inside) draw about 30 mA each. Three of those straight on the pins is too much, so drive each through a transistor:
  ```
  GPIO ──[1 kΩ]── base of NPN (2N2222 / S8050)
  buzzer + ── 3.3 V (or 5 V for a louder buzzer)      buzzer − ── collector
  emitter ── GND
  ```
- **Wires to the ears:** run a twisted pair (signal + GND) to each ear so long wires don't pick up noise. Tape them along the hat band.
- **Active or passive?**
  - An *active* buzzer beeps by itself when powered (often has a sticker on top). Set `ACTIVE_BUZZER = true`. Then every cue has the same pitch, and only *which* buzzer and *how long* differ.
  - A *passive* buzzer needs a tone. Leave it `false`.
  - Not sure? Try `false`. If you only hear clicks, switch to `true`.

  All three buzzers must be the same type.
- Power the ESP32 from the hat's power bank by USB, or from the laptop's USB cable in USB mode.

## 2. Flash the firmware (Arduino IDE)

1. Install the Arduino IDE 2 from https://www.arduino.cc/en/software.
2. Add ESP32 support: **Tools → Board → Boards Manager**, search **"esp32"**, install **"esp32 by Espressif Systems"**.
3. Open `esp32/boxing_buzzer/boxing_buzzer.ino`.
4. Check the settings at the top: `PIN_FRONT` / `PIN_LEFT` / `PIN_RIGHT`, `ACTIVE_BUZZER`, `MOVE_TOWARD_BUZZ`, the WiFi name and password.
5. **Tools → Board → esp32 → "ESP32 Dev Module"** (or your exact board). Then choose **Tools → Port**: the COM port that appears when you plug the board in.
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
| Only clicks, no tone | Set `ACTIVE_BUZZER = true` and re-upload |
| One buzzer silent at power-on | Check that buzzer's wire and GND, and that its pin matches `PIN_FRONT` / `PIN_LEFT` / `PIN_RIGHT` |
| Left and right feel backwards | Swap the two ear wires, or flip `MOVE_TOWARD_BUZZ` |
| Chirps every second | It lost the laptop: the demo crashed or stopped, or the cable/WiFi dropped |
| Too quiet | Use a transistor driver and a bigger buzzer, or put the buzzer closer to the ear |

**Not tested yet:** the firmware hasn't been run on a real ESP32. It was compile-checked with a normal C++ compiler against stand-ins for the ESP32 libraries (both 2.x and 3.x), which catches typos but not hardware behavior. The laptop side was tested against a fake ESP32 over UDP: every cue, heartbeat, ping and goodbye byte arrived correctly.
