"""
One command for the lighting / frame-rate augmentation experiment (meant for a fast machine).

What it does, in order (each stage skips work that is already finished, so it's safe to re-run):
  1. check     Python + package versions, the 4 session videos, disk space, GPU
  2. augment   re-render every session darker+blurred ("dim_blur") and over-exposed ("bright"),
               re-run the pose model on them, and make 15 fps copies   -> data_aug/
  3. features  optical-flow + pixel-change features for every copy     -> features/
  4. pi        same copies with the Pi's ONNX pose model               -> data_onnx_aug/
  5. train     4 cross-validated trainings (laptop/Pi x without/with the copies)
               -> runs/<run-name>/  models, full logs, summary.md

It NEVER overwrites models/ (the live demo's models). Results go to runs/<run-name>/.

  python run_augmented_training.py                  # everything (the real run)
  python run_augmented_training.py --check-only     # just the checks
  python run_augmented_training.py --smoke          # 1-minute test on a tiny slice, writes to smoke_test/
  python run_augmented_training.py --skip-pi        # laptop model only (faster)
"""
import argparse
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

PYTHON_OK = (3, 13)
# The trained models are pickles: they only load reliably on the demo laptop with the SAME
# scikit-learn / numpy versions. These must match exactly.
EXACT = {"scikit-learn": "1.9.1", "numpy": "2.2.6", "joblib": "1.6.0"}
# These should match (small keypoint / flow differences otherwise), but won't break anything.
SHOULD = {"ultralytics": "8.4.216", "opencv-python": "4.13.0.92", "onnxruntime": "1.26.0"}
# Every reviewed session in data/ (each needs its .mp4 locally for this run).
SESSIONS = sorted(p.stem for p in (ROOT / "data").glob("*.npz")
                  if p.with_suffix(".labels.json").exists() and p.with_suffix(".auto_candidates.json").exists())
SMOKE_SOURCE = "session_20261121_095106_097514"   # smallest session, used by --smoke
PRESETS = ["dim_blur", "bright"]
FPS = [15]


def say(msg):
    print(msg, flush=True)


def banner(msg):
    say("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


# ----------------------------------------------------------------------------- 1. check
def check(sessions, data_dir, need_gb, onnx_dir=None):
    banner("1/5 CHECK")
    ok = True
    v = sys.version_info
    if (v.major, v.minor) != PYTHON_OK:
        say(f"  FAIL  Python {v.major}.{v.minor}; need {PYTHON_OK[0]}.{PYTHON_OK[1]}.x")
        ok = False
    else:
        say(f"  ok    Python {platform.python_version()} ({sys.executable})")
    for pkg, want in {**EXACT, **SHOULD}.items():
        try:
            have = md.version(pkg)
        except md.PackageNotFoundError:
            have = None
        if have == want:
            say(f"  ok    {pkg} {have}")
        elif pkg in EXACT:
            say(f"  FAIL  {pkg} {have}, need exactly {want}  (pip install {pkg}=={want})")
            ok = False
        else:
            say(f"  warn  {pkg} {have}, expected {want}  (pip install {pkg}=={want})")
    try:
        import torch
        gpu = torch.cuda.is_available()
        say(f"  info  torch {torch.__version__}; NVIDIA GPU (CUDA): "
            f"{'YES - ' + torch.cuda.get_device_name(0) if gpu else 'no (pose runs on the CPU, slower but fine)'}")
    except Exception as exc:
        say(f"  FAIL  torch not importable: {exc}")
        ok = False
    for name in ("yolo11n-pose.pt", "yolo11n-pose.onnx"):
        if not Path(name).exists():
            say(f"  FAIL  {name} missing (git pull?)")
            ok = False
    import cv2
    import numpy as np
    for stem in sessions:
        npz, mp4 = data_dir / f"{stem}.npz", data_dir / f"{stem}.mp4"
        lab, cand = npz.with_suffix(".labels.json"), npz.with_suffix(".auto_candidates.json")
        miss = [p.name for p in (npz, lab, cand) if not p.exists()]
        if miss:
            say(f"  FAIL  {stem}: missing {miss} (git pull?)")
            ok = False
            continue
        if not mp4.exists():
            say(f"  FAIL  {mp4} missing: get the video from Shawn and put it in {data_dir}/ (exact name)")
            ok = False
            continue
        n_npz = len(np.load(npz)["timestamps"])
        cap = cv2.VideoCapture(str(mp4))
        n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n_vid < n_npz:
            say(f"  FAIL  {mp4.name}: {n_vid} video frames but {n_npz} pose rows. Wrong or truncated file?")
            ok = False
        else:
            say(f"  ok    {mp4.name}: {n_vid} frames ({mp4.stat().st_size / 1e6:.0f} MB)")
        if onnx_dir is not None and not (onnx_dir / f"{stem}_onnx.npz").exists():
            say(f"  FAIL  {onnx_dir}/{stem}_onnx.npz missing: run  python repose_onnx.py {npz}  "
                "(or use --skip-pi)")
            ok = False
    for need in ("data_onnx", "features"):
        if not Path(need).is_dir():
            say(f"  FAIL  folder {need}/ missing (git pull?)")
            ok = False
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < need_gb:
        say(f"  FAIL  only {free:.1f} GB free here; need about {need_gb} GB")
        ok = False
    else:
        say(f"  ok    {free:.0f} GB free (need ~{need_gb} GB)")
    if not ok:
        say("\nSome checks FAILED. Fix them (see TEAMMATE_AUGMENTATION_GUIDE.md) and run again.")
        sys.exit(1)
    say("  all checks passed")


# ----------------------------------------------------------------------------- 2. augment
def augment(sessions, data_dir, out_dir):
    banner("2/5 AUGMENT (lighting copies re-run the pose model; this is the slow part)")
    import augment_lighting as al
    al.OUT = out_dir
    pose = None
    for stem in sessions:
        src = data_dir / f"{stem}.npz"
        for preset in PRESETS:
            if (out_dir / f"{stem}_{preset}.aug.json").exists():
                say(f"  skip  {stem}_{preset} (done)")
                continue
            if pose is None:
                from pose_features import PoseRunner
                pose = PoseRunner()
            t0 = time.time()
            al.augment(src, preset, pose)
            say(f"  done  {stem}_{preset} in {time.time() - t0:.0f}s")
        for f in FPS:
            if (out_dir / f"{stem}_fps{f}.aug.json").exists():
                say(f"  skip  {stem}_fps{f} (done)")
                continue
            al.subsample(src, f)


# ----------------------------------------------------------------------------- 3/4. features + pi
def build_features(folder):
    import early_dataset
    for p in sorted(Path(folder).glob("*.npz")):
        if not p.with_suffix(".aug.json").exists():
            continue
        t0 = time.time()
        early_dataset.build(p)   # computes + caches optical flow and pixel-change features
        say(f"  ok    features for {p.name} ({time.time() - t0:.0f}s)")


def link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def pi_copies(sessions, aug_dir, onnx_dir, out_dir):
    """Same augmented copies, but with keypoints from the Pi's ONNX pose model."""
    banner("4/5 PI COPIES (ONNX pose on the augmented videos)")
    import numpy as np
    out_dir.mkdir(exist_ok=True)
    model = None
    for stem in sessions:
        src_onnx = onnx_dir / f"{stem}_onnx.npz"
        for preset in PRESETS + [f"fps{f}" for f in FPS]:
            a = aug_dir / f"{stem}_{preset}.npz"
            dst = out_dir / f"{stem}_onnx_{preset}.npz"
            if dst.with_suffix(".aug.json").exists():
                say(f"  skip  {dst.stem} (done)")
                continue
            z = dict(np.load(a, allow_pickle=False))
            if preset.startswith("fps"):
                o = np.load(src_onnx)
                keep = z["keep_index"]
                z["keypoints"], z["confidences"] = o["keypoints"][keep], o["confidences"][keep]
            else:
                import cv2
                from repose_onnx import largest
                if model is None:
                    from ultralytics import YOLO
                    model = YOLO("yolo11n-pose.onnx", task="pose")
                cap = cv2.VideoCapture(str(a.with_suffix(".mp4")))
                kp, cf = np.full_like(z["keypoints"], np.nan), np.zeros_like(z["confidences"])
                t0 = time.time()
                for i in range(len(z["timestamps"])):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    kp[i], cf[i] = largest(model(frame, imgsz=320, verbose=False)[0])
                    if i and i % 5000 == 0:
                        say(f"    {dst.stem}: {i}/{len(kp)}")
                cap.release()
                z["keypoints"], z["confidences"] = kp, cf
                say(f"  done  {dst.stem} pose in {time.time() - t0:.0f}s")
            z["pose_model"] = np.array("yolo11n-pose.onnx")
            np.savez_compressed(dst, **z)
            for ext in (".labels.json", ".auto_candidates.json"):
                shutil.copyfile(a.with_suffix(ext), dst.with_suffix(ext))
            link_or_copy(a.with_suffix(".mp4"), dst.with_suffix(".mp4"))
            json.dump(dict(source=src_onnx.name, preset=preset), open(dst.with_suffix(".aug.json"), "w"))


# ----------------------------------------------------------------------------- 5. train
def train(run_dir, data_dir, aug_dir, onnx_dir, onnx_aug_dir, skip_pi, test_mount=None):
    banner("5/5 TRAIN (4 cross-validated trainings; logs in " + str(run_dir) + ")")
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("laptop_baseline", ["--data", str(data_dir), "--aug-data", str(aug_dir)]),
            ("laptop_augmented", ["--data", str(data_dir), "--aug-data", str(aug_dir), "--aug-train"])]
    if not skip_pi:
        jobs += [("pi_baseline", ["--data", str(onnx_dir), "--aug-data", str(onnx_aug_dir)]),
                 ("pi_augmented", ["--data", str(onnx_dir), "--aug-data", str(onnx_aug_dir), "--aug-train"])]
    for name, extra in jobs:
        log = run_dir / f"{name}.log"
        if log.exists() and "saved " in log.read_text(encoding="utf-8", errors="replace"):
            say(f"  skip  {name} (done)")
            continue
        say(f"  run   {name} ... (log: {log})")
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            mount = ["--test-mount", test_mount] if test_mount else []
            r = subprocess.run([sys.executable, "early_detector.py", *extra, *mount, "--budgets", "2",
                                "--save", str(run_dir / f"{name}.joblib")],
                               stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if r.returncode:
            say(f"  FAIL  {name} exited with {r.returncode}; see the end of {log}")
            sys.exit(1)
        say(f"  done  {name} in {(time.time() - t0) / 60:.1f} min")
    summarize(run_dir, [n for n, _ in jobs])


def summarize(run_dir, names):
    rows = []
    for name in names:
        txt = (run_dir / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^learned\s+\|([^|]+)\|\s*(\S+)\s*\|\s*([\d.]+)", txt, re.M)
        overall = f"{m.group(1).split()[3]} caught <=200ms, {m.group(3)} fa/min" if m else "?"
        copies = {k: f"{c}/{n} ({p}%)" + (f", {fa} fa/min" if fa else "")
                  for k, c, n, p, fa in re.findall(
                      r"learned on (\S+)\s+copies: caught <=200ms (\d+)/(\d+) \((\d+)%\)(?:, ([\d.]+) fa/min)?", txt)}
        rows.append((name, overall, copies))
    presets = sorted({k for _, _, c in rows for k in c})
    lines = ["# Augmentation run summary", "",
             f"Machine: {platform.node()} | {platform.platform()} | Python {platform.python_version()}", "",
             "Cross-validated: every number is on footage chunks the model did not train on "
             "(and none of their copies either).", "",
             "| Training | Original footage | " + " | ".join(f"{p} copies" for p in presets) + " |",
             "|---|---|" + "---|" * len(presets)]
    for name, overall, copies in rows:
        lines.append(f"| {name} | {overall} | " + " | ".join(copies.get(p, "-") for p in presets) + " |")
    lines += ["", "Read it like this: *_augmented should catch MORE on the copies (dim_blur / bright / fps15)",
              "than *_baseline, without catching fewer on the original footage at a similar false-alarm rate.",
              "Full output: the .log files next to this one."]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    banner("RESULTS")
    say("\n".join(lines))
    say(f"\nSaved {run_dir / 'summary.md'}")


# ----------------------------------------------------------------------------- smoke test
SMOKE = "smokeslice"   # own name, so its cache files can never collide with a real session's


def make_smoke_slice(stem, src_dir, dst_dir, frames=240):
    """First `frames` frames of a session as a tiny stand-alone session named SMOKE (for --smoke)."""
    import cv2
    import numpy as np
    dst_dir.mkdir(parents=True, exist_ok=True)
    z = dict(np.load(src_dir / f"{stem}.npz", allow_pickle=False))
    n = len(z["timestamps"])
    for k, v in list(z.items()):
        if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n:
            z[k] = v[:frames]
    t_end = z["timestamps"][-1]
    if "commands" in z and len(z["commands"]):
        m = z["command_times"] <= t_end
        z["commands"], z["command_times"] = z["commands"][m], z["command_times"][m]
    np.savez_compressed(dst_dir / f"{SMOKE}.npz", **z)
    lab = json.load(open(src_dir / f"{stem}.labels.json"))
    for k in lab["labelers"]:
        lab["labelers"][k] = [m for m in lab["labelers"][k] if m["departure"] < frames - 20]
    lab.pop("ranges", None)
    json.dump(lab, open(dst_dir / f"{SMOKE}.labels.json", "w"))
    cand = json.load(open(src_dir / f"{stem}.auto_candidates.json"))
    cand["candidates"] = [c for c in cand["candidates"] if c["peak"] < frames - 5]
    json.dump(cand, open(dst_dir / f"{SMOKE}.auto_candidates.json", "w"))
    cap = cv2.VideoCapture(str(src_dir / f"{stem}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = None
    for _ in range(frames):
        ok, f = cap.read()
        if not ok:
            break
        if w is None:
            w = cv2.VideoWriter(str(dst_dir / f"{SMOKE}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                (f.shape[1], f.shape[0]))
        w.write(f)
    w.release()
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true", help="only run the checks")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test (about a minute), in smoke_test/")
    ap.add_argument("--skip-pi", action="store_true", help="skip the Pi (ONNX) copies and trainings")
    ap.add_argument("--test-mount", choices=("head", "desk"),
                    help="score only on sessions recorded with this camera mount (use 'head' once head data exists)")
    ap.add_argument("--run-name", default=time.strftime("aug_%Y%m%d_%H%M"), help="folder name under runs/")
    args = ap.parse_args()

    if args.smoke:
        base = ROOT / "smoke_test"
        if base.exists():
            shutil.rmtree(base)
        data_dir, aug_dir = base / "data", base / "data_aug"
        onnx_dir, onnx_aug_dir = base / "data_onnx", base / "data_onnx_aug"
        make_smoke_slice(SMOKE_SOURCE if SMOKE_SOURCE in SESSIONS else SESSIONS[0], Path("data"), data_dir)
        stem = SMOKE
        sessions = [stem]
        check(sessions, data_dir, need_gb=1)
        augment(sessions, data_dir, aug_dir)
        banner("3/5 FEATURES (smoke)")
        build_features(aug_dir)
        if not args.skip_pi:
            # ONNX version of the tiny slice: re-pose it
            import numpy as np
            from repose_onnx import largest
            from ultralytics import YOLO
            import cv2
            onnx_dir.mkdir(parents=True, exist_ok=True)
            z = dict(np.load(data_dir / f"{stem}.npz"))
            model = YOLO("yolo11n-pose.onnx", task="pose")
            cap = cv2.VideoCapture(str(data_dir / f"{stem}.mp4"))
            for i in range(len(z["timestamps"])):
                ok, f = cap.read()
                z["keypoints"][i], z["confidences"][i] = largest(model(f, imgsz=320, verbose=False)[0])
            cap.release()
            np.savez_compressed(onnx_dir / f"{stem}_onnx.npz", **z)
            for ext in (".labels.json", ".auto_candidates.json", ".mp4"):
                shutil.copyfile(data_dir / f"{stem}{ext}", onnx_dir / f"{stem}_onnx{ext}")
            pi_copies(sessions, aug_dir, onnx_dir, onnx_aug_dir)
            build_features(onnx_aug_dir)
        say("\nSMOKE TEST PASSED (pipeline runs end to end). Training is skipped in the smoke test:")
        say("a 12-second slice has too few punches to train on. Now run without --smoke.")
        for p in Path("features").glob(f"{SMOKE}*"):
            p.unlink()   # remove the tiny slice's feature caches
        shutil.rmtree(base, ignore_errors=True)
        return

    data_dir, aug_dir = Path("data"), Path("data_aug")
    onnx_dir, onnx_aug_dir = Path("data_onnx"), Path("data_onnx_aug")
    check(SESSIONS, data_dir, need_gb=8, onnx_dir=None if args.skip_pi else onnx_dir)
    if args.check_only:
        return
    t_all = time.time()
    augment(SESSIONS, data_dir, aug_dir)
    banner("3/5 FEATURES (optical flow for every copy)")
    build_features(aug_dir)
    if not args.skip_pi:
        pi_copies(SESSIONS, aug_dir, onnx_dir, onnx_aug_dir)
        build_features(onnx_aug_dir)
    train(Path("runs") / args.run_name, data_dir, aug_dir, onnx_dir, onnx_aug_dir, args.skip_pi, args.test_mount)
    say(f"\nALL DONE in {(time.time() - t_all) / 60:.0f} min. Next: commit runs/{args.run_name}/ "
        "(see the guide, step 8).")


if __name__ == "__main__":
    main()
