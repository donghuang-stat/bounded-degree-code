#!/usr/bin/env python3
"""Run a command with a POSIX process-group timeout, without GNU coreutils.

Usage: python tests/timeout_command.py -s TERM 5 COMMAND [ARG ...]
On timeout, return 124 for TERM or 137 for KILL, as GNU timeout does here.
The complete process group receives the signal, including multiprocessing workers.
With --checkpoint-dir, send the signal as soon as --after-checkpoints files exist;
SECONDS is then a safety deadline and a missed checkpoint exits 125.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-s", choices=("TERM", "KILL"), default="TERM")
    ap.add_argument("--checkpoint-dir", type=Path)
    ap.add_argument("--after-checkpoints", type=int, default=1)
    ap.add_argument("seconds", type=float)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if args.seconds <= 0 or args.after_checkpoints < 1 or not args.command:
        ap.error("a positive timeout and a command are required")
    proc = subprocess.Popen(args.command, start_new_session=True)

    def send(sig):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    def interrupted(sig, frame):
        send(sig)
        proc.wait()
        raise SystemExit(128 + sig)

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.checkpoint_dir is not None:
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                rc = proc.poll()
                if rc is not None:
                    return rc if rc >= 0 else 128 - rc
                count = sum(1 for p in args.checkpoint_dir.glob("pair_*.npy")
                            if not p.name.endswith(".tmp.npy"))
                if count >= args.after_checkpoints:
                    send(getattr(signal, "SIG" + args.s))
                    proc.wait()
                    return 124 if args.s == "TERM" else 128 + signal.SIGKILL
                time.sleep(0.01)
            send(signal.SIGKILL)
            proc.wait()
            print("checkpoint condition was not reached before the safety deadline", file=sys.stderr)
            return 125
        rc = proc.wait(timeout=args.seconds)
        return rc if rc >= 0 else 128 - rc
    except subprocess.TimeoutExpired:
        sig = getattr(signal, "SIG" + args.s)
        send(sig)
        proc.wait()
        return 124 if args.s == "TERM" else 128 + sig


if __name__ == "__main__":
    raise SystemExit(main())
