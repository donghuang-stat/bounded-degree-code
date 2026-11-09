"""Standard-library launcher regressions; no scientific calculation or Slurm needed.

Run with ``python3 -m unittest discover -s tests -p 'test_launchers.py'``.
Each test copies the shell launchers into a temporary package, supplies a fake
Python interpreter and, where needed, a fake sbatch.  No public ``out/`` is used.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")

PYTHON_STUB = r'''import json, os, sys
from pathlib import Path

with open(os.environ["MOTIF_STUB_PYTHON_LOG"], "a") as stream:
    stream.write(json.dumps(dict(args=sys.argv[1:], cwd=os.getcwd(),
                                 python=os.environ.get("PYTHON"))) + "\n")
args = sys.argv[1:]
if args and Path(args[0]).name == "run_shard.py":
    if "--list" in args:
        status = int(os.environ.get("MOTIF_STUB_LIST_STATUS", "0"))
        if status:
            sys.exit(status)
        print("3")
    elif "--shard" in args:
        shard = args[args.index("--shard") + 1]
        if shard == os.environ.get("MOTIF_STUB_FAIL_SHARD"):
            sys.exit(7)
'''

SBATCH_STUB = r'''import json, os, sys
from pathlib import Path

args = sys.argv[1:]
Path(os.environ["MOTIF_STUB_SBATCH_LOG"]).write_text(json.dumps(dict(
    args=args, python=os.environ.get("PYTHON"), mode=os.environ.get("MODE"),
    pkg_dir=os.environ.get("PKG_DIR"))))
job = os.environ.get("MOTIF_STUB_JOB_ID", "401")
pattern = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--output="))
numeric_id = job.split(";", 1)[0]
if (os.environ.get("MOTIF_STUB_CREATE_LOG", "1") == "1"
        and numeric_id.isdigit()):
    output = Path(pattern.replace("%j", numeric_id))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("CURRENT_PILOT_LOG_" + job + "\n")
if job:
    print(job, flush=True)
if os.environ.get("MOTIF_STUB_STDERR"):
    print(os.environ["MOTIF_STUB_STDERR"], file=sys.stderr)
sys.exit(int(os.environ.get("MOTIF_STUB_SBATCH_STATUS", "0")))
'''


@unittest.skipUnless(os.name == "posix" and BASH, "POSIX Bash is required")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="motif-launcher-test-")
        self.addCleanup(temporary.cleanup)
        # macOS may expose its temporary directory through a /var symlink;
        # compare physical paths, as the shell's pwd does after changing folders.
        self.root = Path(temporary.name).resolve()
        self.package = self.root / "package with spaces"
        self.caller = self.root / "caller with spaces"
        self.bin_dir = self.root / "stub bin"
        for directory in (self.package / "tests", self.package / "configs",
                          self.package / "slurm", self.caller / "inputs", self.bin_dir):
            directory.mkdir(parents=True, exist_ok=True)
        for relative in ("run_local.sh", "tests/pilot_n5000.sh"):
            shutil.copy2(PACKAGE / relative, self.package / relative)
        # Even a regression that bypasses PYTHON can only execute these guards,
        # never a real graph benchmark or a real scientific module.
        guard = "raise RuntimeError('Scientific execution is forbidden in launcher tests')\n"
        for relative in ("run_shard.py", "bench.py", "slurm/cfginfo.py"):
            (self.package / relative).write_text(guard, encoding="utf-8")
        (self.package / "slurm/run_array.sbatch").write_text("# stub only\n", encoding="utf-8")
        self.config = self.caller / "inputs/config example.json"
        self.config.write_text("{}\n", encoding="utf-8")
        (self.package / "configs/n5000_p05.json").write_text("{}\n", encoding="utf-8")
        self.python_log = self.root / "python-calls.jsonl"
        self.sbatch_log = self.root / "sbatch-call.json"
        self.selected_python = self.root / "selected python with spaces"
        self._write_executable(self.selected_python, PYTHON_STUB)
        self._write_executable(self.bin_dir / "sbatch", SBATCH_STUB)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith("MOTIF_STUB_")}
        self.env.update(PYTHON=str(self.selected_python), PYTHONNOUSERSITE="1",
                        MOTIF_STUB_PYTHON_LOG=str(self.python_log),
                        MOTIF_STUB_SBATCH_LOG=str(self.sbatch_log),
                        PATH=str(self.bin_dir) + os.pathsep + os.environ.get("PATH", ""),
                        CPUS="2", MEM_PER_CPU="5G", SBATCH_EXTRA="")

    @staticmethod
    def _write_executable(path, body):
        # Use the current test interpreter, rather than /usr/bin/env python3.
        path.write_text("#!" + sys.executable + "\n" + body, encoding="utf-8")
        path.chmod(0o755)

    def run_script(self, relative, arguments=(), environment=None):
        if self.python_log.exists():
            self.python_log.unlink()
        env = self.env.copy()
        env.update(environment or {})
        return subprocess.run([BASH, str(self.package / relative)] + list(arguments),
                              cwd=str(self.caller), env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)

    def python_calls(self):
        if not self.python_log.exists():
            return []
        return [json.loads(line) for line in self.python_log.read_text().splitlines()]

    def make_stale_log(self):
        stale = self.package / "out/pilot_n5000/pilot_99999.out"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("STALE_PILOT_LOG_MUST_NOT_BE_USED\n", encoding="utf-8")

    def test_local_resolves_caller_paths_and_honors_selected_python(self):
        result = self.run_script("run_local.sh", ("inputs/config example.json", "3"),
                                 {"PYTHON": "../selected python with spaces"})
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.python_calls()
        self.assertEqual(len(calls), 4)
        for call in calls:
            self.assertEqual(call["cwd"], str(self.package))
            self.assertEqual(call["python"], str(self.selected_python))
            self.assertEqual(call["args"][1], str(self.config))
        self.assertEqual(calls[0]["args"][-1], "--list")
        self.assertEqual([call["args"][call["args"].index("--shard") + 1]
                          for call in calls[1:]], ["0", "1", "2"])
        self.assertTrue(all(call["args"][-2:] == ["--workers", "3"] for call in calls[1:]))

    def test_local_propagates_list_failure_without_running_shards(self):
        result = self.run_script("run_local.sh", (str(self.config),),
                                 {"MOTIF_STUB_LIST_STATUS": "9"})
        self.assertEqual(result.returncode, 9)
        self.assertEqual(len(self.python_calls()), 1)

    def test_local_stops_at_first_failed_shard(self):
        result = self.run_script("run_local.sh", (str(self.config), "2"),
                                 {"MOTIF_STUB_FAIL_SHARD": "1"})
        self.assertEqual(result.returncode, 7)
        calls = self.python_calls()
        self.assertEqual(len(calls), 3)  # --list, shard 0, then failing shard 1
        self.assertEqual(calls[-1]["args"][-3:], ["1", "--workers", "2"])

    def test_local_rejects_bad_usage_before_starting_python(self):
        for arguments in ((), (str(self.config), "0"), (str(self.config), "-1"),
                          (str(self.config), "x"), (str(self.config), "2", "extra")):
            with self.subTest(arguments=arguments):
                result = self.run_script("run_local.sh", arguments)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(self.python_calls(), [])

    def test_pilot_reads_successful_current_job_and_exports_python(self):
        result = self.run_script("tests/pilot_n5000.sh",
                                 environment={"MOTIF_STUB_JOB_ID": "401"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CURRENT_PILOT_LOG_401", result.stdout)
        call = json.loads(self.sbatch_log.read_text())
        self.assertIn("--parsable", call["args"])
        self.assertIn("--wait", call["args"])
        self.assertEqual(call["python"], str(self.selected_python))
        self.assertEqual(call["mode"], "bench")
        self.assertEqual(call["pkg_dir"], str(self.package))
        self.assertEqual(call["args"][-1], str(self.package / "configs/n5000_p05.json"))
        output = next(arg.split("=", 1)[1] for arg in call["args"] if arg.startswith("--output="))
        self.assertTrue(Path(output).is_absolute())

    def test_pilot_accepts_cluster_suffix_and_absolute_config(self):
        result = self.run_script("tests/pilot_n5000.sh", environment={
            "MOTIF_STUB_JOB_ID": "402;cluster-z", "CONFIG": str(self.config)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CURRENT_PILOT_LOG_402;cluster-z", result.stdout)
        self.assertEqual(json.loads(self.sbatch_log.read_text())["args"][-1], str(self.config))

    def test_pilot_preserves_failed_job_status_and_current_log(self):
        self.make_stale_log()
        result = self.run_script("tests/pilot_n5000.sh", environment={
            "MOTIF_STUB_JOB_ID": "403;cluster-z", "MOTIF_STUB_SBATCH_STATUS": "17",
            "MOTIF_STUB_STDERR": "simulated computation failure"})
        self.assertEqual(result.returncode, 17)
        self.assertIn("CURRENT_PILOT_LOG_403;cluster-z", result.stdout)
        self.assertIn("simulated computation failure", result.stderr)
        self.assertNotIn("STALE_PILOT_LOG", result.stdout)

    def test_pilot_rejected_submission_cannot_use_old_log(self):
        self.make_stale_log()
        result = self.run_script("tests/pilot_n5000.sh", environment={
            "MOTIF_STUB_JOB_ID": "", "MOTIF_STUB_SBATCH_STATUS": "6",
            "MOTIF_STUB_STDERR": "simulated submission rejection"})
        self.assertEqual(result.returncode, 6)
        self.assertIn("simulated submission rejection", result.stderr)
        self.assertNotIn("STALE_PILOT_LOG", result.stdout)

    def test_pilot_missing_current_log_cannot_use_old_log(self):
        self.make_stale_log()
        result = self.run_script("tests/pilot_n5000.sh", environment={
            "MOTIF_STUB_JOB_ID": "404", "MOTIF_STUB_CREATE_LOG": "0"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pilot_404.out", result.stderr)
        self.assertNotIn("STALE_PILOT_LOG", result.stdout)

    def test_pilot_invalid_job_id_cannot_use_old_log(self):
        self.make_stale_log()
        result = self.run_script("tests/pilot_n5000.sh", environment={
            "MOTIF_STUB_JOB_ID": "invalid-job-id", "MOTIF_STUB_CREATE_LOG": "0"})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("STALE_PILOT_LOG", result.stdout)

    def test_pilot_without_slurm_uses_selected_python_for_local_fallback(self):
        # Restrict PATH to the harmless utilities needed by the shell wrapper:
        # a machine running these tests can never discover its real sbatch here.
        fallback_bin = self.root / "no Slurm bin"
        fallback_bin.mkdir()
        for command in ("dirname", "basename", "mkdir"):
            (fallback_bin / command).symlink_to(shutil.which(command))
        result = self.run_script("tests/pilot_n5000.sh", environment={
            "PATH": str(fallback_bin), "CONFIG": str(self.config)})
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.python_calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["args"][0], "slurm/cfginfo.py")
        self.assertEqual(calls[1]["args"][0], "bench.py")
        self.assertTrue(all(call["python"] == str(self.selected_python) for call in calls))


if __name__ == "__main__":
    unittest.main()
