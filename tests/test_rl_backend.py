import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires the PyTorch evaluation runtime")
class RLBackendTest(unittest.TestCase):
    def test_unified_runner_evaluates_checkpoint_after_legacy_modules_are_removed(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            command = [
                sys.executable, str(root / "scripts/run_seed_benchmark.py"),
                "--methods", "rl", "--scenarios", "sa_bt",
                "--configurations", "kn9_ks3_km20_kb3", "--seeds", "0",
                "--limit", "1", "--rl-device", "cpu",
                "--scenario-dir", str(folder / "scenarios"),
                "--output-dir", str(folder / "results"),
            ]
            first = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            output = folder / "results/sa_bt/kn9_ks3_km20_kb3/runs.csv"
            rows = pd.read_csv(output)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows.method.item(), "RL(s.10)")
            self.assertEqual(rows.decoder.item(), "policy")
            self.assertIn(rows.feasible.item(), (0, 1))
            self.assertEqual(rows.solver_seed.item(), 0)
            self.assertTrue((folder / "results/table.tex").is_file())
            before = output.read_bytes()
            second = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
