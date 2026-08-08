"""Regenerate paper-neurips/tables/tab_cross_model.tex from
transfer_eval_<target>.json files written by cross_model_transfer_eval.py.

For each scenario in {S1, S3a} we emit a 3x3 grid of (attacker x evaluator)
ASR cells at clean / opus64 / opus32; same-model cells (white-box) are
filled from the existing results_codec_robust bundle.

Each cell is $p_{\\pm h}$ with Wilson 95% half-width in pp. ASR is computed
by exact target-substring match in the model output (no WER fallback) so
the metric is consistent with the cross-scenario tables.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EVAL_ROLLUPS_DIR, PROJECT_ROOT

ROOT = Path(PROJECT_ROOT)
# Reference rollups ship in data/eval_rollups/; generated .tex goes to results/tables/.
ROLLUPS = Path(EVAL_ROLLUPS_DIR)
TABLES_OUT = ROOT / "results" / "tables"
RCR = ROLLUPS
OUT = TABLES_OUT / "tab_cross_model.tex"

MODELS = [
    ("qwen2_audio",    "Qwen2-Audio"),
    ("qwen25_omni",    "Qwen2.5-Omni"),
    ("audio_flamingo", "Audio Flamingo 3"),
]
CHANNELS = ["clean", "opus64", "opus32"]
CHAN_LABEL = {"clean": "Clean", "opus64": "Opus 64k", "opus32": "Opus 32k"}

# Headline (attacker bundle, channel) per scenario.
# Each scenario uses a single eps for the cross-model row to keep the table compact.
SCENARIOS = [
    ("S1",  "01_finance_voice_agent",   1.0, ""),
    ("S3a", "03a_ai_detection_bypass",  1.0, "_multibitrate"),
]

Z_95 = 1.959963984540054
_DECODER = json.JSONDecoder()
_PAIR_SUFFIX_RE = re.compile(r"_pair\d+_summary\.json$")
S1_BASE_VOICES = {"aria", "christopher", "eric", "michelle", "roger"}


def wilson_half(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    p = k / n
    denom = 1.0 + Z_95 * Z_95 / n
    half = Z_95 * math.sqrt(p * (1.0 - p) / n + Z_95 * Z_95 / (4.0 * n * n)) / denom
    center = (p + Z_95 * Z_95 / (2.0 * n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return 100.0 * (hi - lo) / 2.0


def cell(k: int, n: int) -> str:
    if n <= 0:
        return r"\pending"
    return f"${100.0 * k / n:.1f}_{{\\pm {wilson_half(k, n):.0f}}}$"


def _load_lenient(path: Path):
    txt = path.read_text().lstrip()
    if not txt:
        return None
    try:
        obj, _ = _DECODER.raw_decode(txt)
        return obj
    except json.JSONDecodeError:
        return None


def _keep_for(scen_key: str, path: Path) -> bool:
    """Same filters as build_main_results_tables.py: drop S3 strays, restrict
    S1 to the 5-voice intersection so cells are comparable across attackers."""
    if scen_key in ("S3a", "S3b") and not _PAIR_SUFFIX_RE.search(path.name):
        return False
    if scen_key == "S1":
        if not path.name.startswith("music_banking_"):
            return False
        voice = path.name.split("_", 3)[2]
        return voice in S1_BASE_VOICES
    return True


def _whitebox_cell(scen_key: str, scen_dir: str, suffix: str,
                   attacker: str, eps: float, channel: str):
    """For attacker == evaluator we read the existing _summary.json /
    _codecs.json files (no transfer eval needed)."""
    bundle = RCR / scen_dir / attacker / f"eps_{eps:.1f}{suffix}" / "audio"
    if not bundle.is_dir():
        return None
    summs = sorted(bundle.glob("*_summary.json"))
    summs = [s for s in summs if _keep_for(scen_key, s)]
    k = n = 0
    for s in summs:
        d = _load_lenient(s)
        if d is None:
            continue
        target = (d.get("target_text") or "").strip().lower()
        if not target:
            continue
        if channel == "clean":
            out = (d.get("final_output") or "").strip().lower()
        else:
            br = int(channel.replace("opus", ""))
            row = (d.get("opus_robustness") or {}).get(str(br)) or {}
            out = (row.get("output") or "").strip().lower()
            if not row:
                continue
        n += 1
        if target in out:
            k += 1
    return (k, n)


def _transfer_cell(scen_dir: str, suffix: str, attacker: str, evaluator: str,
                   eps: float, channel: str, scen_key: str):
    """Read transfer_eval_<evaluator>.json from the attacker's bundle dir."""
    bundle = RCR / scen_dir / attacker / f"eps_{eps:.1f}{suffix}"
    f = bundle / f"transfer_eval_{evaluator}.json"
    if not f.is_file():
        return None
    d = _load_lenient(f)
    if d is None:
        return None
    k = n = 0
    for r in d.get("results", []):
        # apply same per-cell filter to the clip stem
        clip = r.get("clip", "")
        # build a fake summary path so _keep_for can use the filename
        fake = bundle / "audio" / f"{clip}_summary.json"
        if not _keep_for(scen_key, fake):
            continue
        target = (r.get("target_text") or "").strip().lower()
        if not target:
            continue
        ch = r.get(channel) or {}
        out = (ch.get("output") or "").strip().lower()
        if not ch:
            continue
        n += 1
        if target in out:
            k += 1
    return (k, n)


def build_table() -> str:
    lines = [
        r"%% ---- Cross-model transfer (attacker x evaluator) ----",
        r"\begin{table}[t]",
        r"    \centering",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \caption{\textbf{Cross-model transfer.} ASR (\%, target-substring "
        r"match) when the adversarial wav optimized against an \emph{attacker} "
        r"model is fed to a different \emph{evaluator} model. Diagonal cells are "
        r"the white-box result (same wavs, same model). Each cell is $p_{\pm h}$ "
        r"with $h$ the 95\% Wilson half-width (pp). Headline eps per scenario.}",
        r"    \label{tab:cross_model}",
        r"    \begin{tabular}{lll ccc}",
        r"    \toprule",
        r"    \textbf{Scenario} & \textbf{Attacker} & \textbf{Channel} & "
        r"\textbf{Eval: Qwen2-Audio} & \textbf{Eval: Qwen2.5-Omni} & "
        r"\textbf{Eval: Audio Flamingo 3} \\",
        r"    \midrule",
    ]
    for scen_idx, (scen_key, scen_dir, eps, suffix) in enumerate(SCENARIOS):
        if scen_idx > 0:
            lines.append(r"    \midrule")
        scen_label = f"{scen_key} ($\\epsilon{{=}}${eps})"
        lines.append(r"    \multirow{9}{*}{" + scen_label + "}")
        for atk_i, (atk_key, atk_label) in enumerate(MODELS):
            if atk_i > 0:
                lines.append(r"    \cmidrule(lr){2-6}")
            for ch_i, ch in enumerate(CHANNELS):
                cells = []
                for ev_key, _ev_label in MODELS:
                    if ev_key == atk_key:
                        kn = _whitebox_cell(scen_key, scen_dir, suffix,
                                            atk_key, eps, ch)
                    else:
                        kn = _transfer_cell(scen_dir, suffix, atk_key, ev_key,
                                            eps, ch, scen_key)
                    cells.append(cell(*kn) if kn else r"\pending")
                atk_cell = r"\multirow{3}{*}{" + atk_label + "}" if ch_i == 0 else ""
                lines.append("      & " + " & ".join([atk_cell, CHAN_LABEL[ch]] + cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    tex = build_table()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
