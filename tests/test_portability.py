#!/usr/bin/env python3
"""Reporting and POSIX orchestration regressions; no large graph computations.

Run: python -m unittest discover -s tests -p 'test_portability.py' -v
"""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench
import campaign_plot


class ReportingTests(unittest.TestCase):
    def test_family_labels_match_aggregate(self):
        from aggregate import families
        for lo, hi in ((3, 3), (3, 4), (3, 5), (4, 4), (3, 8)):
            with self.subTest(lo=lo, hi=hi):
                plotted = [f[0] for f in campaign_plot.plot_families(lo, hi)]
                available = {f[0] for f in families(None, lo, hi)}
                self.assertEqual(len(plotted), len(set(plotted)))
                self.assertTrue(set(plotted) <= available)
                self.assertEqual(len(plotted), 4 if lo + 1 < hi else 3)
        self.assertEqual(campaign_plot.family_label("ALL", 3, 3), "ALL:3")

    def test_rss_units_on_linux_and_macos(self):
        self.assertEqual(bench.peak_rss_gib(2**30, "darwin"), 1.0)
        self.assertEqual(bench.peak_rss_gib(2**20, "linux"), 1.0)

    def test_singleton_campaign_renders_existing_summary(self):
        with tempfile.TemporaryDirectory(prefix="motif-report-") as tmp:
            tmp = Path(tmp)
            out = tmp / "config-out"
            analysis = out / "analysis"
            analysis.mkdir(parents=True)
            cfg = dict(name="singleton", n=6, p=0.2, mmin=3, mmax=3,
                       rhos=[0.0, 0.8], reps=[4, 4], seed=2026, shard_size=4,
                       out_dir=str(out), cache_dir=str(tmp / "cache"))
            config = tmp / "config.json"
            config.write_text(json.dumps(cfg))
            campaign = tmp / "campaign.json"
            campaign.write_text(json.dumps(dict(name="singleton", title="Singleton regression",
                                               out_dir=str(tmp / "campaign-out"),
                                               configs=[str(config)])))
            rows = [dict(family=f[0], n=6, p=0.2, rho=0.8, typeI=0.05,
                         auc=0.7, auc_lo=0.6, auc_hi=0.8, auc_theory=0.75,
                         power=0.3, power_lo=0.2, power_hi=0.4)
                    for f in campaign_plot.plot_families(3, 3)]
            summary = analysis / "summary.csv"
            with summary.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            before = summary.read_bytes()
            result = subprocess.run([sys.executable, str(ROOT / "campaign_plot.py"), str(campaign)],
                                    cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(summary.read_bytes(), before)
            generated = tmp / "campaign-out" / "analysis"
            self.assertGreater((generated / "auc_vs_rho_all_p.png").stat().st_size, 1000)
            self.assertGreater((generated / "power_vs_rho_all_p.pdf").stat().st_size, 1000)
            self.assertIn("ALL:3", (generated / "campaign_numbers.txt").read_text())


@unittest.skipUnless(os.name == "posix", "shell/SLURM tests require POSIX")
class TimeoutTests(unittest.TestCase):
    def run_timeout(self, sig, script, seconds="0.5"):
        return subprocess.run([sys.executable, str(ROOT / "tests/timeout_command.py"),
                               "-s", sig, seconds, sys.executable, "-c", script],
                              text=True, capture_output=True, timeout=10)

    def test_child_failure_is_preserved(self):
        self.assertEqual(self.run_timeout("TERM", "raise SystemExit(7)").returncode, 7)

    def test_term_reaches_child_and_allows_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            code = ("import signal,time,pathlib; "
                    f"signal.signal(signal.SIGTERM,lambda s,f:(pathlib.Path({str(checkpoint)!r})"
                    ".write_text('saved'),exit(0))); time.sleep(30)")
            result = self.run_timeout("TERM", code)
            self.assertEqual(result.returncode, 124, result.stderr)
            self.assertEqual(checkpoint.read_text(), "saved")

    def test_kill_reaches_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            heartbeat = Path(tmp) / "heartbeat"
            child = ("import pathlib,time; p=pathlib.Path(" + repr(str(heartbeat)) + "); "
                     "exec('while True:\\n p.write_text(str(time.time_ns()))\\n time.sleep(.03)')")
            parent = ("import subprocess,sys,time; "
                      f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)")
            result = self.run_timeout("KILL", parent, seconds="0.8")
            self.assertEqual(result.returncode, 128 + signal.SIGKILL, result.stderr)
            before = heartbeat.read_text()
            # A second short subprocess supplies a delay without importing time here.
            subprocess.run([sys.executable, "-c", "import time; time.sleep(.15)"], check=True)
            self.assertEqual(heartbeat.read_text(), before)

    def test_checkpoint_trigger_ignores_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "pair_0_0.npy"
            code = ("import pathlib,time; p=pathlib.Path(" + repr(tmp) + "); "
                    "(p/'pair_0_0.tmp.npy').write_text('partial'); time.sleep(.2); "
                    "(p/'pair_0_0.npy').write_text('complete'); time.sleep(30)")
            result = subprocess.run([sys.executable, str(ROOT / "tests/timeout_command.py"),
                                     "-s", "TERM", "--checkpoint-dir", tmp, "3",
                                     sys.executable, "-c", code],
                                    text=True, capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 124, result.stderr)
            self.assertEqual(checkpoint.read_text(), "complete")

    def test_missing_checkpoint_fails_and_stops_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / "tests/timeout_command.py"),
                                     "--checkpoint-dir", tmp, "0.1", sys.executable,
                                     "-c", "import time; time.sleep(30)"],
                                    text=True, capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 125)
            self.assertIn("checkpoint condition was not reached", result.stderr)


@unittest.skipUnless(os.name == "posix", "mock SLURM requires POSIX")
class MockSlurmTests(unittest.TestCase):
    def test_isolated_state_and_wait_failure(self):
        with tempfile.TemporaryDirectory(prefix="motif-mock-test-") as tmp:
            tmp = Path(tmp)
            envs = [dict(os.environ, MOCK_SLURM_DIR=str(tmp / name), PYTHON=sys.executable)
                    for name in ("one", "two")]

            def call(command, arguments, env):
                return subprocess.run([sys.executable, str(ROOT / "tests/mock_slurm/mockslurm.py"),
                                       command] + arguments, cwd=tmp, env=env,
                                      text=True, capture_output=True, timeout=10)

            try:
                for env in envs:
                    result = call("sbatch", ["--parsable", "--job-name=isolated_monitor",
                                              "--wrap=sleep 30"], env)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), "1000")
                call("cleanup", [], envs[0])
                self.assertEqual(call("squeue", [], envs[0]).stdout, "")
                self.assertIn("isolated_monitor", call("squeue", [], envs[1]).stdout)
                failed = call("sbatch", ["--parsable", "--wait", "--wrap=exit 7"], envs[0])
                self.assertEqual(failed.returncode, 7, failed.stderr)
                self.assertEqual(failed.stdout.strip(), "1001")
                queued = call("sbatch", ["--parsable", "--wrap=exit 7"], envs[0])
                self.assertEqual(queued.returncode, 0)
                self.assertEqual(queued.stdout.strip(), "1002")
                state = json.loads((tmp / "two" / "jobs.json").read_text())
                self.assertEqual(state["next"], 1001)
                self.assertEqual(list(state["jobs"]), ["1000"])
            finally:
                for env in envs:
                    call("cleanup", [], env)


if __name__ == "__main__":
    unittest.main()
