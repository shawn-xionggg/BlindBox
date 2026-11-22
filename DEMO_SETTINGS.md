# Demo settings that worked live (2026-11-22)

**Tested live, works well:** puncher starts from a still guard, throws one punch, returns to guard.

```powershell
python dodge_live.py --source 1 --model models/archive/TESTED_live_leaveguard_headonly.joblib --threshold 0.82
```

| Setting | Value | Why |
|---|---|---|
| Model | `models/archive/TESTED_live_leaveguard_headonly.joblib` | Hat-camera sessions only, start = fist leaves guard (`make_reactive_labels.py`, `data_reactive/`). Offline on people it never saw: 40% caught within 200 ms of leaving guard at its strict setting, 52% looser |
| Threshold | **0.82** | Lower than the saved 0.922 / 0.909: the demo starts every punch from a still guard, so a lower threshold beeps earlier and more often, and false beeps are cheap (an unneeded dodge) |
| Camera | `--source 1` (C270) | Check `python -c "import cv2; print([cv2.VideoCapture(i, cv2.CAP_DSHOW).isOpened() for i in range(3)])"` if it won't open |
| Frame rate | aim for **~30 fps** | at 17 fps every frame is 59 ms apart and the beep comes ~50 ms later. Light the puncher, laptop plugged in, Windows "Best performance" |

Tried and worse (kept in `models/candidates/`): firing within 2 or 3 frames instead of 4 (`--pos-frames`), mirrored copies, "toward target" labels for the reactive detector.

**Not done yet:** this is the laptop (PyTorch pose) model. The Pi runs ONNX pose and a C model: the same training on `data_onnx` copies (`repose_onnx.py`) and `export_model_c.py` are still needed for the Pi demo (HEAD_MOUNTED_TRAINING.md section 6). The installed `models/early_detector*.joblib` are unchanged.
