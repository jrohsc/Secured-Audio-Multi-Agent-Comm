"""
Emit a LaTeX table for S1 finance-voice-agent in the *non-codec-robust*
baseline (channel-mode=none, no Opus EOT during training).

Mirrors the 18-column layout of paper-neurips/tables/tab_S1.tex:
  Model | eps | n | Clean | Opus(16/24/32/64/128/192) | MP3(64/96/128/192)
        | AAC(64/96/128/192)

Cells use strict substring match (target_text in output, lowercased), to
match build_s1_codec_latex.py and build_main_results_tables.py.
Confidence is Wilson 95% half-width (pp). Missing cells emit \\pending.

Auto-discovers every `eps_<x>` bundle under
results_codec_non_robust/01_finance_voice_agent/<model>/, so the same
script can be re-run as more attacks finish.

Usage:
    python scripts/build_s1_non_robust_table.py
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
# The non-robust (no-EoT) bundles are not redistributed; regenerate them with
# run_benchmark.py, or pass --bundle-parent at a path of your own.
BUNDLE_PARENT = ROLLUPS / "01_finance_voice_agent"
DEFAULT_OUT = TABLES_OUT / "tab_S1_non_robust.tex"

OPUS_BRS = [16, 24, 32, 64, 128, 192]
MP3_BRS  = [64, 96, 128, 192]
AAC_BRS  = [64, 96, 128, 192]
N_TARGET = 50  # full S1 manifest

Z_95 = 1.959963984540054

# Print order matches tab_S1.tex
MODEL_ORDER = ["qwen2_audio", "qwen25_omni", "audio_flamingo"]
MODEL_LABEL = {
    "qwen2_audio": "Qwen2-Audio",
    "qwen25_omni": "Qwen2.5-Omni",
    "audio_flamingo": "Audio Flamingo 3",
}


def _wilson_half_pp(k: int, n: int) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1.0 + Z_95 * Z_95 / n
    half = Z_95 * math.sqrt(p * (1.0 - p) / n + Z_95 * Z_95 / (4.0 * n * n)) / denom
    return 100.0 * half


def _fmt_cell(k: int, n: int) -> str:
    if n <= 0:
        return r"\pending"
    pct = 100.0 * k / n
    half = _wilson_half_pp(k, n)
    return f"${pct:5.1f}_{{\\pm {half:.0f}}}$"


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def scan_bundle(bundle: Path) -> dict | None:
    """Return per-channel (hits, n) tuples from a bundle dir, or None."""
    audio = bundle / "audio"
    summs = sorted(audio.glob("*_summary.json"))
    if not summs:
        return None
    n_used = 0
    clean_hit = 0
    opus = {br: [0, 0] for br in OPUS_BRS}
    mp3  = {br: [0, 0] for br in MP3_BRS}
    aac  = {br: [0, 0] for br in AAC_BRS}
    for s in summs:
        d = _load_json(s)
        if d is None:
            continue
        tgt = (d.get("target_text") or "").strip().lower()
        if not tgt:
            continue
        n_used += 1
        fo = (d.get("final_output") or "").strip().lower()
        if tgt in fo:
            clean_hit += 1
        for br_str, v in (d.get("opus_robustness") or {}).items():
            try:
                br = int(br_str)
            except ValueError:
                continue
            if br not in opus or not isinstance(v, dict):
                continue
            out = (v.get("output") or "").strip().lower()
            opus[br][1] += 1
            if tgt in out:
                opus[br][0] += 1
        c = s.with_name(s.name.replace("_summary.json", "_codecs.json"))
        dc = _load_json(c) if c.exists() else None
        if dc is None:
            continue
        for fam, bucket, allowed in [
            ("mp3_robustness", mp3, MP3_BRS),
            ("aac_robustness", aac, AAC_BRS),
        ]:
            for br_str, v in (dc.get(fam) or {}).items():
                try:
                    br = int(br_str)
                except ValueError:
                    continue
                if br not in allowed or not isinstance(v, dict):
                    continue
                out = (v.get("output") or "").strip().lower()
                bucket[br][1] += 1
                if tgt in out:
                    bucket[br][0] += 1
    return {
        "n": n_used,
        "clean": (clean_hit, n_used),
        "opus": {br: tuple(v) for br, v in opus.items()},
        "mp3":  {br: tuple(v) for br, v in mp3.items()},
        "aac":  {br: tuple(v) for br, v in aac.items()},
    }


def fmt_n(n_actual: int, n_target: int) -> str:
    if n_actual == n_target:
        return f"{n_actual}"
    return f"{n_actual}/{n_target}"


def discover_eps(model_dir: Path) -> list[tuple[float, str]]:
    """Return (eps_value, dirname) pairs sorted by eps_value."""
    out: list[tuple[float, str]] = []
    if not model_dir.is_dir():
        return out
    for sub in sorted(model_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = re.match(r"^eps_([0-9]+(?:\.[0-9]+)?)$", sub.name)
        if not m:
            continue
        out.append((float(m.group(1)), sub.name))
    out.sort(key=lambda t: t[0])
    return out


def row_for(model: str, eps: float, eps_dir_name: str) -> str | None:
    bundle = BUNDLE_PARENT / model / eps_dir_name
    r = scan_bundle(bundle)
    if r is None:
        return None
    n_str = fmt_n(r["n"], N_TARGET)
    cells = [_fmt_cell(*r["clean"])]
    for br in OPUS_BRS:
        cells.append(_fmt_cell(*r["opus"][br]))
    for br in MP3_BRS:
        cells.append(_fmt_cell(*r["mp3"][br]))
    for br in AAC_BRS:
        cells.append(_fmt_cell(*r["aac"][br]))
    eps_str = f"{eps:.1f}"
    return f"& {eps_str} & {n_str} & " + " & ".join(cells) + r" \\"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    n_opus = len(OPUS_BRS)
    n_mp3  = len(MP3_BRS)
    n_aac  = len(AAC_BRS)
    # cols: Model, eps, n, Clean, Opus*6, MP3*4, AAC*4 = 4 + 14 = 18
    n_total = 1 + 1 + 1 + 1 + n_opus + n_mp3 + n_aac
    colspec = "ll c c " + "c" * n_opus + " " + "c" * n_mp3 + " " + "c" * n_aac

    # multicol header column ranges (1-indexed)
    c0 = 5  # first Opus column (after Model, eps, n, Clean)
    c_opus = (c0, c0 + n_opus - 1)
    c_mp3  = (c_opus[1] + 1, c_opus[1] + n_mp3)
    c_aac  = (c_mp3[1] + 1, c_mp3[1] + n_aac)

    def hdr(brs):
        return " & ".join(f"\\textbf{{{b}k}}" for b in brs)

    lines: list[str] = []
    lines.append("% Auto-generated by scripts/build_s1_non_robust_table.py")
    lines.append("% Source: results_codec_non_robust/01_finance_voice_agent/")
    lines.append("% Strict substring match; Wilson 95% half-width in pp.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \scriptsize")
    lines.append(r"    \setlength{\tabcolsep}{3pt}")
    lines.append(
        r"    \caption{\textbf{S1 finance voice-agent --- non-codec-robust baseline.} "
        r"Same scenario, carriers, and target model as Table~\ref{tab:cross_s1}, but "
        r"trained \emph{without} the multi-bitrate Opus EOT step (channel-mode=none). "
        r"Cells are $p_{\pm h}$ with $h$ the 95\% Wilson half-width (pp); "
        r"target-substring match against the model's response. "
        r"Held-out MP3 and AAC bitrates are inference-only (no retraining). "
        r"\textbf{Headline:} clean attack succeeds, but every Opus / MP3 / AAC bitrate "
        r"degrades sharply versus the codec-robust attack --- quantifying the price "
        r"of dropping codec EOT from training.}"
    )
    lines.append(r"    \label{tab:s1_non_robust}")
    lines.append(r"    \resizebox{\textwidth}{!}{")
    lines.append(r"    \begin{tabular}{" + colspec + r"}")
    lines.append(r"    \toprule")
    lines.append(
        f"     & & & & \\multicolumn{{{n_opus}}}{{c}}{{\\textbf{{Opus (held-out)}}}} "
        f"& \\multicolumn{{{n_mp3}}}{{c}}{{\\textbf{{MP3 (held-out)}}}} "
        f"& \\multicolumn{{{n_aac}}}{{c}}{{\\textbf{{AAC-LC (held-out)}}}} \\\\"
    )
    lines.append(
        f"    \\cmidrule(lr){{{c_opus[0]}-{c_opus[1]}}} "
        f"\\cmidrule(lr){{{c_mp3[0]}-{c_mp3[1]}}} "
        f"\\cmidrule(lr){{{c_aac[0]}-{c_aac[1]}}}"
    )
    lines.append(
        r"    \textbf{Model} & \textbf{$\epsilon$} & \textbf{$n$} & \textbf{Clean} & "
        + hdr(OPUS_BRS) + " & "
        + hdr(MP3_BRS)  + " & "
        + hdr(AAC_BRS)  + r" \\"
    )
    lines.append(r"    \midrule")

    first_model = True
    for model in MODEL_ORDER:
        model_dir = BUNDLE_PARENT / model
        eps_list = discover_eps(model_dir)
        rows: list[tuple[float, str]] = []
        for eps, dname in eps_list:
            row = row_for(model, eps, dname)
            if row is not None:
                rows.append((eps, row))
        if not rows:
            continue
        if not first_model:
            lines.append(r"    \cmidrule(lr){1-" + str(n_total) + "}")
        first_model = False
        label = MODEL_LABEL[model]
        prefix = f"    \\multirow{{{len(rows)}}}{{*}}{{{label}}} "
        for i, (_eps, row) in enumerate(rows):
            if i == 0:
                lines.append(prefix + row)
            else:
                lines.append("                                  " + row)

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    }")
    lines.append(r"\end{table*}")

    out = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out)
    print(out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
