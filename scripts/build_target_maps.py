"""Build per-scenario `_target_map.json` by tracing each symlinked wav back to its
source `config.json` and extracting the matching target string.

Run from the repo root:
    python scripts/build_target_maps.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SCENARIOS = Path("scenarios")


def _find_source_config(symlink: Path) -> Path | None:
    """Walk up from the symlink target to find the nearest config.json."""
    real = symlink.resolve()
    for d in [real.parent, real.parent.parent, real.parent.parent.parent]:
        cfg = d / "config.json"
        if cfg.is_file():
            return cfg
    return None


def _try_per_file_summary(symlink: Path) -> tuple[str | None, dict]:
    """Bank/interview style: each clip has its own `<stem>_summary.json` containing target_text."""
    real = symlink.resolve()
    stem = real.stem
    # Try the wav's actual stem and a few common manglings (strip _attacked suffix, etc.)
    candidates = [stem, re.sub(r"_attacked$", "", stem),
                  re.sub(r"_eps\d+(_attacked)?$", "", stem)]
    for s in candidates:
        for sib in [real.parent / f"{s}_summary.json",
                    real.parent.parent / f"{s}_summary.json"]:
            if sib.is_file():
                try:
                    j = json.loads(sib.read_text())
                except Exception:
                    continue
                tgt = j.get("target_text") or j.get("target")
                if tgt:
                    return tgt, {"matched_summary": str(sib)}
    return None, {}


def _find_target(real_name: str, cfg_path: Path) -> tuple[str | None, dict]:
    """Look up a per-file target in cfg_path:targets/commands by name heuristic."""
    cfg = json.loads(cfg_path.read_text())
    targets = cfg.get("targets") or cfg.get("commands") or {}
    if not isinstance(targets, dict):
        return None, {"note": "targets is not dict"}
    # Direct key in stem
    stem = Path(real_name).stem
    for key in targets:
        if key in stem:
            return targets[key], {"matched_key": key}
    # Try clip_id pattern e.g. "chinese_magicdata_m_q1_idx0"
    m = re.search(r"(chinese|interview|saycan|nav|comm|emerg|sys|nego)_[\w]+", stem)
    if m and m.group(0) in targets:
        return targets[m.group(0)], {"matched_key": m.group(0)}
    # Some interview/bank summaries embed eps_attacked suffix; strip it
    cleaned = re.sub(r"_eps[\d]+(_attacked)?$", "", stem)
    if cleaned in targets:
        return targets[cleaned], {"matched_key": cleaned, "note": "eps_attacked stripped"}
    return None, {"note": "no match", "looked_up_stem": stem}


def build_for_scenario(scenario_dir: Path) -> dict[str, str]:
    target_map: dict[str, str] = {}
    misses: list[dict] = []
    audio_root = scenario_dir / "audio"
    if not audio_root.exists():
        return target_map
    for wav in audio_root.rglob("*.wav"):
        if not wav.is_symlink():
            continue
        # Try per-file *_summary.json first (bank/interview pattern)
        tgt, info = _try_per_file_summary(wav)
        if tgt is None:
            cfg = _find_source_config(wav)
            if cfg is None:
                misses.append({"wav": wav.name, "reason": "no source config.json or _summary.json"})
                continue
            tgt, info = _find_target(wav.resolve().name, cfg)
            if tgt is None:
                misses.append({"wav": wav.name, "info": info, "config": str(cfg)})
                continue
        target_map[wav.name] = tgt
    miss_path = scenario_dir / "_target_map.misses.json"
    if misses:
        miss_path.write_text(json.dumps(misses, indent=2))
    elif miss_path.exists():
        miss_path.unlink()
    return target_map


def main() -> None:
    for scenario_dir in sorted(SCENARIOS.iterdir()):
        if not (scenario_dir / "scenario.json").is_file():
            continue
        tmap = build_for_scenario(scenario_dir)
        out = scenario_dir / "_target_map.json"
        out.write_text(json.dumps(tmap, indent=2))
        miss = scenario_dir / "_target_map.misses.json"
        miss_n = len(json.loads(miss.read_text())) if miss.is_file() else 0
        print(f"  {scenario_dir.name:48} matched {len(tmap):5d} | missed {miss_n:5d}")


if __name__ == "__main__":
    main()
