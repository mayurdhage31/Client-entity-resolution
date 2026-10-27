"""Check the public Git file set and immutable portfolio artifacts without ML imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FREEZE = "reports/test_evaluation/v1"
INDIA = "outputs/india_demo/v1"
BLOCKED_PARTS = {".venv", "venv", ".cache", "__pycache__", ".aws", ".codex", ".agents", ".secrets"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"),
    "OpenAI-style key": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "credential in URL": re.compile(r"https?://[^\s/:<>]+:[^\s/@<>]+@"),
    "private machine path": re.compile(r"(?i)(?:\b[A-Z]:[\\/](?:Users|Documents and Settings)[\\/]|/(?:Users|home)/)[A-Za-z0-9_.-]+[\\/]"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def project_path(relative):
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in relative,
            f"Invalid manifest path: {relative}")
    path = ROOT.joinpath(*pure.parts)
    require(path.resolve().is_relative_to(ROOT), f"Path leaves project: {relative}")
    return path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(relative):
    return json.loads(project_path(relative).read_text(encoding="utf-8-sig"))


def verify_artifacts():
    expected = {}

    def add_hash(relative, checksum):
        require(re.fullmatch(r"[0-9a-f]{64}", checksum), f"Invalid checksum: {relative}")
        require(relative not in expected or expected[relative] == checksum,
                f"Manifests disagree: {relative}")
        expected[relative] = checksum

    freeze = read_json(f"{FREEZE}/freeze_manifest.json")
    freeze_hash = project_path(f"{FREEZE}/freeze_manifest.sha256").read_text().strip()
    add_hash(f"{FREEZE}/freeze_manifest.json", freeze_hash)
    for relative, checksum in freeze["protected_sha256"].items():
        add_hash(relative, checksum)

    # Include the original completion commit and the saved reproducibility rerun.
    for commit_path in sorted((ROOT/FREEZE).rglob("artifacts.json")):
        relative_dir = commit_path.parent.relative_to(ROOT/FREEZE).as_posix()
        commit = json.loads(commit_path.read_text(encoding="utf-8-sig"))
        require(commit["freeze_sha256"] == freeze_hash, f"Wrong freeze in {relative_dir}")
        for name, checksum in commit["report_sha256"].items():
            add_hash(f"{FREEZE}/{relative_dir}/{name}", checksum)
        for name, checksum in commit["output_sha256"].items():
            add_hash(f"outputs/test_evaluation/v1/{relative_dir}/{name}", checksum)

    run = read_json(f"{INDIA}/prediction_run.json")
    require(run["original_freeze_sha256"] == freeze_hash, "Indian demo freeze disagrees")
    require(run["model_sha256"] == freeze["selected_model_sha256"], "Indian demo model disagrees")
    require(run["threshold"] == freeze["threshold"], "Indian demo threshold disagrees")
    for key in ("data_sha256", "prediction_files_sha256", "frozen_input_sha256"):
        for relative, checksum in run[key].items():
            add_hash(relative, checksum)
    add_hash("build_india_demo.py", run["builder_sha256"])
    for relative, checksum in expected.items():
        path = project_path(relative)
        require(path.is_file(), f"Missing frozen artifact: {relative}")
        require(digest(path) == checksum, f"Changed frozen artifact: {relative}")
    return expected


def publication_files():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, capture_output=True, check=True,
    )
    return sorted(set(name.decode("utf-8") for name in result.stdout.split(b"\0") if name))


def screen_files(files):
    issues, total_bytes = [], 0
    for relative in files:
        path = project_path(relative)
        parts = PurePosixPath(relative).parts
        name = path.name.lower()
        credential_file = (
            (name.startswith(".env") and name not in {".env.example", ".env.sample", ".env.template"})
            or name.startswith("credentials") or name.startswith("secrets.")
            or name.endswith((".pem", ".key", ".p12", ".pfx", ".safetensors"))
        )
        if BLOCKED_PARTS.intersection(parts) or credential_file:
            issues.append(f"Excluded/private file in publication set: {relative}")
            continue
        require(path.is_file(), f"Publication file missing: {relative}")
        size = path.stat().st_size
        total_bytes += size
        if size >= 50 * 1024 * 1024:
            issues.append(f"File at least 50 MiB; review storage before publication: {relative}")
            continue
        raw = path.read_bytes()
        if b"\0" in raw or path.suffix.lower() == ".joblib":
            continue
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                # Do not print any matched secret values.
                issues.append(f"Possible {label}: {relative}")
    require(not issues, "\n".join(issues))
    return total_bytes


def main():
    expected = verify_artifacts()
    files = publication_files()
    require(files, "No publishable files found; initialize Git in the project root")
    require(set(expected) <= set(files), "A required frozen artifact is excluded from Git")
    total_bytes = screen_files(files)
    print(f"PASS: {len(expected)} saved artifact checksums match")
    print(f"PASS: {len(files)} publishable files, {total_bytes / (1024 * 1024):.2f} MiB total")
    print("PASS: no screened credential patterns, private paths, cache files or oversized files")
    print("Screening uses a limited pattern set; review new files before committing.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
