"""
Regression tests for the learned early detector. They use the recorded,
reviewed sessions in data/ and skip if those (or gcc) aren't present.

  python -m unittest test_early
"""
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SESSION = HERE / "data" / "session_20261121_024335_748065.npz"
HAVE_DATA = SESSION.exists() and SESSION.with_suffix(".labels.json").exists() and \
    SESSION.with_suffix(".auto_candidates.json").exists() and (HERE / "models" / "early_detector.joblib").exists()
GCC = shutil.which("gcc") or (r"C:\msys64\ucrt64\bin\gcc.exe" if Path(r"C:\msys64\ucrt64\bin\gcc.exe").exists() else None)


@unittest.skipUnless(HAVE_DATA, "recorded session / model not available")
class EarlyDetectorTests(unittest.TestCase):
    def test_live_features_match_training_features(self):
        r = subprocess.run([sys.executable, "test_early_live.py", str(SESSION), "--frames", "400"],
                           cwd=HERE, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("match offline features exactly", r.stdout)

    def test_model_scale_guard(self):
        import early_live
        import joblib
        b = joblib.load(HERE / "models" / "early_detector.joblib")
        self.assertEqual(float(b.get("flow_scale", 0.5)), early_live.SCALE)

    def test_detector_replay_produces_cues(self):
        r = subprocess.run([sys.executable, "dodge_live.py", "--source", str(SESSION.with_suffix(".mp4")),
                            "--npz", str(SESSION), "--headless"], cwd=HERE, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("cues", r.stdout)

    @unittest.skipUnless(GCC, "gcc not available")
    def test_exported_c_model_matches_python(self):
        env = dict(os.environ)
        env["PATH"] = str(Path(GCC).parent) + os.pathsep + env.get("PATH", "")
        r = subprocess.run([sys.executable, "export_model_c.py"], cwd=HERE, capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("matches scikit-learn", r.stdout)


    @unittest.skipUnless(GCC and (HERE / "yolo11n-pose.onnx").exists(), "gcc or ONNX model not available")
    def test_c_pose_decoder_matches_ultralytics(self):
        env = dict(os.environ)
        env["PATH"] = str(Path(GCC).parent) + os.pathsep + env.get("PATH", "")
        r = subprocess.run([sys.executable, "check_pose_decode.py", "--frames", "60"], cwd=HERE,
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("matches ultralytics", r.stdout)


if __name__ == "__main__":
    unittest.main()
