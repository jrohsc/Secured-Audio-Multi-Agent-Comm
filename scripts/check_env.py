#!/usr/bin/env python3
"""
Smoke check for a fresh clone.

Verifies, without loading any model weights or running an attack, that:

  1. every .py in the repo compiles,
  2. every module each script imports is actually resolvable,
  3. the shipped data directories the drivers read are present,
  4. the third-party packages the attack needs are installed,
  5. ffmpeg is reachable (needed for every Opus / MP3 / AAC channel).

Run it from anywhere:

    python scripts/check_env.py

Exit status is 0 when everything required is present, 1 otherwise. Optional
pieces (Kimi-Audio, AudioSeal, PESQ, the not-redistributed music carriers)
are reported as warnings and do not fail the run.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import py_compile
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
WRAPPERS = REPO / "codec_wrappers"

# Import targets that are genuinely optional: they gate a single experiment and
# the rest of the repo runs without them.
OPTIONAL_MODULES = {
    "dac",              # descript-audio-codec  -> DAC attacks only
    "snac",             # snac                  -> SNAC wrapper only
    "encodec",          # encodec               -> EnCodec attacks only
    "audioseal",        # AudioSeal defense
    "pesq", "pyloudnorm", "jiwer", "noisereduce", "sounddevice",
    "edge_tts",         # only to regenerate TTS carriers
    "kimia_infer",      # Kimi-Audio, needs its own conda env
    "moshi",            # PersonaPLEX / Moshi
    "matplotlib", "pandas", "seaborn",
}

# Data the drivers read. Missing copyrighted carriers are expected.
REQUIRED_DIRS = [
    "data/speech",
    "data/music/ai_generated",
    "data/eval_rollups",
    "codec_wrappers/models",
]
OPTIONAL_DIRS = [
    "data/music/copyrighted",   # 9 commercial recordings, not redistributed
]

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = RESET = ""


def ok(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET}  {msg}")


def bad(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{RESET}  {msg}")


def python_files() -> list[Path]:
    files: list[Path] = []
    for base in (SCRIPTS, WRAPPERS):
        files.extend(sorted(base.rglob("*.py")))
    return files


def check_compile(files: list[Path]) -> int:
    print("\n[1/5] Byte-compiling every source file")
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(files):
            try:
                py_compile.compile(
                    str(f), doraise=True, cfile=str(Path(tmp) / f"{i}.pyc")
                )
            except py_compile.PyCompileError as e:
                bad(f"{f.relative_to(REPO)}: {e.msg.strip().splitlines()[-1]}")
                failures += 1
    if not failures:
        ok(f"{len(files)} files compile")
    return failures


def top_level_imports(path: Path) -> set[str]:
    """Root module name of every import in the file, including nested ones."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import; it resolves within the package.
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def check_imports(files: list[Path]) -> int:
    print("\n[2/5] Resolving every imported module")
    # Mirror the sys.path the drivers build via config.
    for p in (str(SCRIPTS), str(WRAPPERS)):
        if p not in sys.path:
            sys.path.insert(0, p)

    missing: dict[str, list[str]] = {}
    optional_missing: set[str] = set()
    for f in files:
        for name in sorted(top_level_imports(f)):
            if name in sys.builtin_module_names:
                continue
            try:
                found = importlib.util.find_spec(name) is not None
            except (ImportError, ValueError):
                found = False
            if found:
                continue
            if name in OPTIONAL_MODULES:
                optional_missing.add(name)
            else:
                missing.setdefault(name, []).append(
                    str(f.relative_to(REPO))
                )

    for name, users in sorted(missing.items()):
        bad(f"'{name}' not importable — needed by {', '.join(sorted(set(users))[:3])}")
    if optional_missing:
        warn("optional, install only for the matching experiment: "
             + ", ".join(sorted(optional_missing)))
    if not missing:
        ok("all required imports resolve")
    return len(missing)


def check_config() -> int:
    print("\n[3/5] Loading config and checking derived paths")
    try:
        import config
    except Exception as e:  # pragma: no cover - diagnostic path
        bad(f"config.py failed to import: {type(e).__name__}: {e}")
        return 1

    failures = 0
    if Path(config.PROJECT_ROOT) != REPO:
        bad(f"PROJECT_ROOT is {config.PROJECT_ROOT}, expected {REPO}")
        failures += 1
    else:
        ok(f"PROJECT_ROOT -> {config.PROJECT_ROOT}")

    if not Path(config.MUSIC_DIR).is_dir():
        bad(f"MUSIC_DIR does not exist: {config.MUSIC_DIR}")
        failures += 1
    else:
        ok(f"MUSIC_DIR -> {config.MUSIC_DIR}")

    for name, path in sorted(config.MODEL_PATHS.items()):
        if "${" in str(path):
            bad(f"MODEL_PATHS[{name!r}] still holds an unexpanded placeholder: {path}")
            failures += 1
    if not failures:
        ok("MODEL_PATHS fully resolved (env var or HuggingFace repo id)")
    return failures


def check_data() -> int:
    print("\n[4/5] Checking shipped data directories")
    failures = 0
    for rel in REQUIRED_DIRS:
        if (REPO / rel).is_dir():
            ok(rel)
        else:
            bad(f"missing required directory: {rel}")
            failures += 1
    for rel in OPTIONAL_DIRS:
        if not (REPO / rel).is_dir():
            warn(f"{rel} absent — S3b needs the 9 carriers listed in "
                 "docs/MANIFEST_AUDIO.md")
    return failures


def check_tools() -> int:
    print("\n[5/5] Checking external tools")
    import config
    ffmpeg = config.FFMPEG_BIN
    if os.path.isfile(ffmpeg) or shutil.which(ffmpeg):
        ok(f"ffmpeg -> {ffmpeg}")
        return 0
    bad("ffmpeg not found; every Opus / MP3 / AAC channel needs it "
        "(conda install -c conda-forge ffmpeg)")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    print(f"CodecAttack environment check\nrepo: {REPO}\npython: {sys.version.split()[0]}")
    files = python_files()
    failures = (
        check_compile(files)
        + check_imports(files)
        + check_config()
        + check_data()
        + check_tools()
    )

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET} See docs/REPRODUCE.md for setup.")
        return 1
    print(f"{GREEN}All checks passed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
