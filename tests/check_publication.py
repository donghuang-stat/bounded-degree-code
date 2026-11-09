#!/usr/bin/env python3
"""Audit only the candidate files in a portable publication allowlist.

No Git command or network call is made.  Pass ``--file-list FILE`` to audit an
explicit tracked/release inventory (newline or NUL separated; ``-`` means stdin).
Without an inventory, local build/output directories are skipped and the source
tree is checked against the allowlist.  A tracked cache is still rejected when
it appears in an explicit inventory.  Findings name files and rules, never the
contents of a credential.  This checks known signatures, not every possible
kind of confidential information.
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import zipfile
import zlib


ROOT_FILES = {
    "README.md", "LICENSE", "NOTICE", "CITATION.cff", "CITATION.bib",
    "requirements.txt", "requirements-figures.txt", "requirements-tested-py312.txt",
    "THIRD_PARTY_NOTICES.md", "environment.yml", "pyproject.toml", "run_local.sh",
    ".gitignore", ".gitattributes", "PUBLICATION_FILES.txt", "CHECKSUMS.sha256",
    "aggregate.py", "bench.py", "campaign_plot.py", "graphs.py", "monitor.py",
    "motiflib.py", "plots.py", "run_shard.py", "roc.py", "plot_roc.py", "validate_roc.py",
    "signal_growth.py", "plot_paper.py",
}
SLURM_FILES = {"cfginfo.py", "common.sh", "resubmit_missing.sh", "run_array.sbatch",
               "submit.sh", "submit_campaign.sh"}
TEST_SHELL_FILES = {"pilot_n5000.sh", "test_local_campaign.sh", "test_slurm_small.sh"}
MOCK_FILES = {"mockslurm.py", "sbatch", "squeue", "scancel"}
DATA_SUFFIXES = {".csv", ".json", ".npz", ".png", ".pdf", ".md", ".txt", ".sha256"}
LOCAL_ONLY = {".git", "out", "cache", "__pycache__", ".pytest_cache", ".mypy_cache",
              ".ruff_cache", ".venv", "venv"}
GENERATED_TOP_LEVEL = {"figures"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".pkl", ".pickle", ".pem", ".key", ".zip", ".gz"}
MAX_EXPANDED_BYTES = 64 * 1024 * 1024

# Generic signatures contain no actual local identity or account.  The path
# prefix is split where necessary so a pattern does not match its own source.
SIGNATURES = [
    ("private local account", re.compile(rb"\ba6s[0-9]{6,}\b")),
    ("cluster account", re.compile(rb"BSCC-[A-Za-z0-9]+")),
    ("personal workstation path", re.compile(rb"/Us" + rb"ers/[A-Za-z0-9._-]+")),
    ("private cluster path", re.compile(rb"/public[0-9]+/(?:home|soft)(?:/|\b)")),
    ("development-home path", re.compile(rb"/ho" + rb"me/[A-Za-z0-9._-]+/(?:work|Desktop|Documents)/")),
    ("private cluster host", re.compile(rb"(?:para" + rb"cloud\.com|\.para\.bscc)")),
    ("SSH private-key block", re.compile(rb"-----BE" + rb"GIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("GitHub credential", re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}")),
]


def permitted(relative):
    parts = PurePosixPath(relative).parts
    if not parts or any(p in (".", "..") for p in parts):
        return False
    name = parts[-1]
    lower = name.lower()
    if any(p in (".ssh", ".aws", ".codex", "site", "job_logs") for p in parts):
        return False
    if (lower in ("ssh_config", "connection.md", ".env", ".ds_store")
            or lower.startswith(("id_rsa", "id_ed25519", "credentials"))
            or "private_key" in lower or Path(name).suffix.lower() in FORBIDDEN_SUFFIXES):
        return False
    if len(parts) == 1:
        return name in ROOT_FILES
    if parts[0] in ("configs", "campaigns"):
        return Path(name).suffix in (".json", ".md")
    if parts[0] == "slurm":
        return len(parts) == 2 and name in SLURM_FILES
    if parts[0] == "tests":
        if len(parts) == 3 and parts[1] == "mock_slurm":
            return name in MOCK_FILES
        return len(parts) == 2 and (name in TEST_SHELL_FILES or name.endswith(".py"))
    if parts[0] in ("data", "examples"):
        if any(p in ("logs", "progress") for p in parts):
            return False
        if parts[0] == "examples" and lower in ("manifest.json", "status.json", "progress.log"):
            return False
        return Path(name).suffix.lower() in DATA_SUFFIXES
    if parts[0] == "docs":
        return (Path(name).suffix == ".md" or
                (len(parts) == 3 and parts[1] == "figures" and
                 Path(name).suffix.lower() in (".png", ".pdf")))
    if parts[0] == ".github":
        return Path(name).suffix in (".yml", ".yaml", ".md")
    return False


def default_inventory(root):
    for directory, subdirs, files in os.walk(str(root), followlinks=False):
        subdirs[:] = sorted(name for name in subdirs if name not in LOCAL_ONLY and
                            not (Path(directory) == root and name in GENERATED_TOP_LEVEL))
        for name in sorted(files):
            yield (Path(directory) / name).relative_to(root).as_posix()


def text_blocks(path, payload):
    """Expose compressed scientific-data/figure metadata without executing it."""
    yield payload
    # Unicode NumPy strings and UTF-16 metadata may interleave ASCII with NULs.
    if b"\x00" in payload:
        yield payload.replace(b"\x00", b"")
    if path.suffix.lower() == ".npz":
        with zipfile.ZipFile(path) as archive:
            expanded = sum(info.file_size for info in archive.infolist())
            if expanded > MAX_EXPANDED_BYTES:
                raise ValueError("compressed array exceeds expanded-size limit")
            for info in archive.infolist():
                if not info.filename.endswith(".npy"):
                    raise ValueError("unexpected member in scientific-data archive")
                member = archive.read(info)
                yield member
                yield member.replace(b"\x00", b"")
    if path.suffix.lower() == ".png" and payload.startswith(b"\x89PNG\r\n\x1a\n"):
        offset = 8
        while offset + 12 <= len(payload):
            size = struct.unpack(">I", payload[offset:offset + 4])[0]
            kind = payload[offset + 4:offset + 8]
            body = payload[offset + 8:offset + 8 + size]
            if len(body) != size:
                raise ValueError("truncated PNG chunk")
            if kind == b"zTXt":
                _, compressed = body.split(b"\x00", 1)
                if not compressed or compressed[0] != 0:
                    raise ValueError("unsupported compressed PNG metadata")
                value = zlib.decompress(compressed[1:])
                if len(value) > MAX_EXPANDED_BYTES:
                    raise ValueError("PNG metadata exceeds expanded-size limit")
                yield value
            elif kind == b"iTXt":
                # keyword, flag/method, language, translated keyword, text
                _, rest = body.split(b"\x00", 1)
                flag, method = rest[0], rest[1]
                _, translated = rest[2:].split(b"\x00", 1)
                _, value = translated.split(b"\x00", 1)
                if flag:
                    if method != 0:
                        raise ValueError("unsupported compressed PNG metadata")
                    value = zlib.decompress(value)
                if len(value) > MAX_EXPANDED_BYTES:
                    raise ValueError("PNG metadata exceeds expanded-size limit")
                yield value
            offset += size + 12


def audit(root, inventory, max_file_bytes):
    findings = []
    total = 0
    checked = 0
    seen = set()
    for relative in inventory:
        relative = relative.strip()
        if not relative or relative.startswith("#"):
            continue
        if relative in seen:
            findings.append(dict(path=relative, rule="duplicate inventory entry"))
            continue
        seen.add(relative)
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "\\" in relative or not permitted(relative):
            findings.append(dict(path=relative, rule="file is outside the publication allowlist"))
            continue
        path = root.joinpath(*pure.parts)
        if any(parent.is_symlink() for parent in [path] + list(path.parents) if parent != root):
            findings.append(dict(path=relative, rule="symlink is not a publication source file"))
            continue
        if not path.is_file():
            findings.append(dict(path=relative, rule="inventory file is missing"))
            continue
        size = path.stat().st_size
        total += size
        if size > max_file_bytes:
            findings.append(dict(path=relative, rule="file exceeds reviewed size limit", bytes=size))
            continue
        checked += 1
        try:
            payload = path.read_bytes()
            matched = set()
            for block in text_blocks(path, payload):
                for label, signature in SIGNATURES:
                    if label not in matched and signature.search(block):
                        findings.append(dict(path=relative, rule=label))
                        matched.add(label)
        except (OSError, ValueError, IndexError, zipfile.BadZipFile, zlib.error) as exc:
            findings.append(dict(path=relative, rule="could not inspect file metadata",
                                 error=type(exc).__name__))
    if total > MAX_EXPANDED_BYTES:
        findings.append(dict(path=".", rule="candidate package exceeds reviewed total-size limit", bytes=total))
    return dict(passed=not findings, files_checked=checked, candidate_files=len(seen),
                total_bytes=total, findings=findings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--file-list", help="tracked/release paths, newline or NUL separated; '-' reads stdin")
    parser.add_argument("--max-file-mib", type=float, default=8.0)
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")
    args = parser.parse_args()
    if args.max_file_mib <= 0:
        parser.error("--max-file-mib must be positive")
    root = args.root.resolve()
    if args.file_list:
        text = sys.stdin.read() if args.file_list == "-" else Path(args.file_list).read_text()
        inventory = text.split("\x00") if "\x00" in text else text.splitlines()
    else:
        inventory = default_inventory(root)
    report = audit(root, inventory, int(args.max_file_mib * 1024 * 1024))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for finding in report["findings"]:
            print("FAIL {}: {}".format(finding["path"], finding["rule"]))
        print("Publication audit {}: {} files, {:,} bytes".format(
            "PASSED" if report["passed"] else "FAILED", report["files_checked"], report["total_bytes"]))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
