"""
TensorFlow Lite version of the pose model (plan B runtime for QNX).

Converted with onnx2tf from the verified yolo11n-pose.onnx (ultralytics blocks
TFLite export on Windows). Compares TFLite vs ONNX Runtime on real letterboxed
frames -- raw output and decoded keypoints (same decoder as qnx/pose/pose_decode.c)
-- and times TFLite with 1 and 4 threads.

  python check_tflite_pose.py [video.mp4] [--frames 100] [--model yolo11n-pose_tflite/yolo11n-pose_float32.tflite]
"""
import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from ultralytics.data.augment import LetterBox

from check_opencv_dnn import decode_largest


def tflite_runner(path, threads):
    it = tf.lite.Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]

    def run(x_nchw):
        x = x_nchw
        if list(inp["shape"]) == [1, 320, 320, 3]:
            x = np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1))
        it.set_tensor(inp["index"], x.astype(inp["dtype"]))
        it.invoke()
        y = it.get_tensor(out["index"])[0]
        return y if y.shape[0] == 56 else y.T          # -> (56, N)
    return run, inp["shape"], out["shape"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="data/session_20261121_030328_194149.mp4")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--model", default="yolo11n-pose_tflite/yolo11n-pose_float32.tflite")
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, args.frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    lb = LetterBox(new_shape=(320, 320), auto=False, stride=32)
    sess = ort.InferenceSession("yolo11n-pose.onnx", providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    run, ishape, oshape = tflite_runner(args.model, 4)
    print(f"{args.model}: input {list(ishape)}, output {list(oshape)}")
    raw, kpd, visd, mism = [], [], [], 0
    xs = []
    for f in frames:
        x = np.ascontiguousarray(lb(image=f)[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        xs.append(x)
        a = sess.run(None, {name: x})[0][0]
        b = run(x)
        raw.append(np.abs(a - b).max())
        da, db = decode_largest(a, f.shape[1], f.shape[0]), decode_largest(b, f.shape[1], f.shape[0])
        if (da is None) != (db is None):
            mism += 1
        elif da is not None:
            kpd.append(np.abs(da[0] - db[0]).max())
            visd.append(np.abs(da[1] - db[1]).max())
    print(f"raw output max |diff| vs ONNX Runtime: median {np.median(raw):.2e}, worst {max(raw):.2e}")
    print(f"decoded keypoints: found/not mismatches {mism}; max diff median {np.median(kpd):.3f} px, "
          f"worst {max(kpd):.3f} px; visibility worst {max(visd):.4f}")
    for threads in (1, 4):
        r, _, _ = tflite_runner(args.model, threads)
        for x in xs[:3]:
            r(x)
        t0 = time.perf_counter()
        for x in xs[:50]:
            r(x)
        print(f"TFLite {threads} thread(s): {(time.perf_counter() - t0) / min(50, len(xs)) * 1000:.1f} ms/frame on this laptop")


if __name__ == "__main__":
    main()
