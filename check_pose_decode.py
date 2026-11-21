"""
Verify the plain-C pose decoder (qnx/pose/pose_decode.c) against ultralytics.

Raw ONNX Runtime outputs for real frames are written to a binary file, decoded
by the C code (compiled with gcc), and compared with ultralytics' own decoding
of the same ONNX model: same person found, keypoints within a tiny tolerance.

  python check_pose_decode.py [video.mp4] [--frames 200]
"""
import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

HERE = Path(__file__).parent
POSE = HERE / "qnx" / "pose"

TEST_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include "pose_decode.h"
int main(int argc, char **argv) {
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb"); if (!f) return 2;
    int frames, n, w, h; static float out[56 * 2100]; static int scratch[2 * 2100];
    if (fread(&frames, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) return 3;
    for (int i = 0; i < frames; i++) {
        if (fread(&w, 4, 1, f) != 1 || fread(&h, 4, 1, f) != 1) return 4;
        if (fread(out, 4, (size_t)(56 * n), f) != (size_t)(56 * n)) return 5;
        pose_person_t p;
        letterbox_t lb = letterbox_params(w, h, 320);
        if (!pose_decode_largest(out, n, lb, 0.25f, 0.7f, scratch, &p)) { printf("none\n"); continue; }
        for (int k = 0; k < POSE_KPTS; k++) printf("%.4f %.4f %.4f ", p.kpt[k][0], p.kpt[k][1], p.kconf[k]);
        printf("\n");
    }
    fclose(f);
    return 0;
}
"""


def largest(r):
    if r.keypoints is None or r.boxes is None or not len(r.boxes):
        return None
    b = r.boxes.xyxy.cpu().numpy()
    i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    return np.c_[r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=str(HERE / "data" / "session_20261121_030328_194149.mp4"))
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, n_all - 1, args.frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            frames.append(f)

    sess = ort.InferenceSession(str(HERE / "yolo11n-pose.onnx"), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    lbx = LetterBox(new_shape=(320, 320), auto=False, stride=32)
    bin_path = POSE / "test_outputs.bin"
    with open(bin_path, "wb") as fh:
        n = None
        chunks = []
        for f in frames:
            img = lbx(image=f)[:, :, ::-1].transpose(2, 0, 1)
            x = np.ascontiguousarray(img, dtype=np.float32)[None] / 255.0
            out = sess.run(None, {name: x})[0][0]          # (56, N)
            n = out.shape[1]
            chunks.append((f.shape[1], f.shape[0], out.astype(np.float32)))
        fh.write(np.array([len(chunks), n], np.int32).tobytes())
        for w, h, out in chunks:
            fh.write(np.array([w, h], np.int32).tobytes())
            fh.write(np.ascontiguousarray(out).tobytes())

    (POSE / "test_decode.c").write_text(TEST_C)
    exe = POSE / ("test_decode.exe" if os.name == "nt" else "test_decode")
    env = dict(os.environ)
    if os.name == "nt":
        env["PATH"] = r"C:\msys64\ucrt64\bin;" + env.get("PATH", "")
    subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(POSE), "-o", str(exe),
                    str(POSE / "test_decode.c"), str(POSE / "pose_decode.c"), "-lm"], check=True, env=env)
    res = subprocess.run([str(exe), str(bin_path)], capture_output=True, text=True, check=True, env=env)
    c_lines = res.stdout.strip().splitlines()

    ref = YOLO(str(HERE / "yolo11n-pose.onnx"), task="pose")
    diffs, cdiffs, mismatch = [], [], 0
    for f, line in zip(frames, c_lines):
        r = largest(ref(f, imgsz=320, verbose=False)[0])
        if line == "none" or r is None:
            mismatch += (line == "none") != (r is None)
            continue
        c = np.array(line.split(), float).reshape(17, 3)
        diffs.append(np.abs(c[:, :2] - r[:, :2]).max())
        cdiffs.append(np.abs(c[:, 2] - r[:, 2]).max())
    print(f"{len(c_lines)} frames | person found/not-found disagreements: {mismatch}")
    print(f"max keypoint difference C vs ultralytics: {max(diffs):.4f} px; max visibility difference {max(cdiffs):.5f}")
    if mismatch or max(diffs) > 0.05 or max(cdiffs) > 1e-3:
        raise SystemExit("MISMATCH")
    print("OK: C pose decoder matches ultralytics")


if __name__ == "__main__":
    main()
