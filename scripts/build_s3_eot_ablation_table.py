"""
Emit a 2-row LaTeX ablation table (codec-robust EOT vs no-EOT non-robust)
for one (scenario, model, eps) cell, mirroring the column layout and red-
shading convention of paper-neurips/tables/tab_S3_eps_1.0.tex.

Default: S3a / Qwen2-Audio / eps=1.0.

Cells: target-substring match (%), no Wilson CI (matches tab_S3 main-body
style; CIs live in the appendix). Cells >= shade_threshold are wrapped in
\\cellcolor{red!10}.

Usage:
    python scripts/build_s3_eot_ablation_table.py
"""
from __future__ import annotations

import argparse
import json
import glob
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EVAL_ROLLUPS_DIR, PROJECT_ROOT

ROOT = Path(PROJECT_ROOT)
# Reference rollups ship in data/eval_rollups/; generated .tex goes to results/tables/.
ROLLUPS = Path(EVAL_ROLLUPS_DIR)
TABLES_OUT = ROOT / "results" / "tables"
RCR = ROLLUPS
# Non-robust (no-EoT) arm is regenerated, not shipped.
RNR = ROLLUPS
DEFAULT_OUT = TABLES_OUT / "tab_S3a_eot_ablation_eps_1.0.tex"

OPUS_BRS = ["16", "24", "32", "64", "128", "192"]
MP3_BRS  = ["64", "96", "128", "192"]
AAC_BRS  = ["64", "96", "128", "192"]


def aggregate(audio_dir: Path) -> dict | None:
    if not audio_dir.is_dir():
        return None
    summs = sorted(audio_dir.glob("*_summary.json"))
    if not summs:
        return None
    n_used = 0
    clean = 0
    opus = {b: [0, 0] for b in OPUS_BRS}
    mp3  = {b: [0, 0] for b in MP3_BRS}
    aac  = {b: [0, 0] for b in AAC_BRS}
    for s in summs:
        try:
            d = json.loads(s.read_text())
        except Exception:
            continue
        tgt = (d.get("target_text") or "").strip().lower()
        if not tgt:
            continue
        n_used += 1
        fo = (d.get("final_output") or "").strip().lower()
        if tgt in fo:
            clean += 1
        for b, v in (d.get("opus_robustness") or {}).items():
            if b not in opus or not isinstance(v, dict):
                continue
            out = (v.get("output") or "").strip().lower()
            opus[b][1] += 1
            if tgt in out:
                opus[b][0] += 1
        c = s.with_name(s.name.replace("_summary.json", "_codecs.json"))
        if not c.exists():
            continue
        try:
            dc = json.loads(c.read_text())
        except Exception:
            continue
        for fam, bucket, allowed in [
            ("mp3_robustness", mp3, MP3_BRS),
            ("aac_robustness", aac, AAC_BRS),
        ]:
            for b, v in (dc.get(fam) or {}).items():
                if b not in allowed or not isinstance(v, dict):
                    continue
                out = (v.get("output") or "").strip().lower()
                bucket[b][1] += 1
                if tgt in out:
                    bucket[b][0] += 1
    return {
        "n": n_used,
        "clean": (clean, n_used),
        "opus": {b: tuple(v) for b, v in opus.items()},
        "mp3":  {b: tuple(v) for b, v in mp3.items()},
        "aac":  {b: tuple(v) for b, v in aac.items()},
    }


def fmt_cell(k: int, n: int, shade_thresh: float) -> str:
    if n <= 0:
        return r"\pending"
    pct = 100.0 * k / n
    s = f"{pct:.1f}"
    if pct >= shade_thresh:
        return r"\cellcolor{red!10}" + s
    return s


def build_row(label: str, agg: dict | None, shade_thresh: float) -> str:
    if agg is None:
        cells = [r"\pending"] * (1 + len(OPUS_BRS) + len(MP3_BRS) + len(AAC_BRS))
        return f"{label} & " + " & ".join(cells) + r" \\"
    cells = [fmt_cell(*agg["clean"], shade_thresh)]
    for b in OPUS_BRS:
        cells.append(fmt_cell(*agg["opus"][b], shade_thresh))
    for b in MP3_BRS:
        cells.append(fmt_cell(*agg["mp3"][b], shade_thresh))
    for b in AAC_BRS:
        cells.append(fmt_cell(*agg["aac"][b], shade_thresh))
    return f"{label} & " + " & ".join(cells) + r" \\"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario-dir", default="03a_ai_detection_bypass")
    p.add_argument("--scenario-label", default="S3a")
    p.add_argument("--model", default="qwen2_audio")
    p.add_argument("--model-label", default="Qwen2-Audio")
    p.add_argument("--eps", default="1.0")
    p.add_argument("--robust-suffix", default="_multibitrate",
                   help="Suffix on results_codec_robust eps dir (e.g. '_multibitrate').")
    p.add_argument("--shade-threshold", type=float, default=80.0,
                   help="Cells >= this percentage get red!10 shading.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    eps_dir = f"eps_{args.eps}"
    robust_dir = RCR / args.scenario_dir / args.model / (eps_dir + args.robust_suffix) / "audio"
    nonrobust_dir = RNR / args.scenario_dir / args.model / eps_dir / "audio"

    rob = aggregate(robust_dir)
    nrb = aggregate(nonrobust_dir)

    rob_n = rob["n"] if rob else 0
    nrb_n = nrb["n"] if nrb else 0
    n_str = f"{rob_n}" if rob_n == nrb_n else f"{rob_n}\\,/\\,{nrb_n}"

    n_total_cols = 1 + 1 + 1 + 1 + len(OPUS_BRS) + len(MP3_BRS) + len(AAC_BRS)
    # Cols: Scenario, Model, Training, Clean, Opus*6, MP3*4, AAC*4 = 4 + 14 = 18
    colspec = "lll c " + "c" * len(OPUS_BRS) + " " + "c" * len(MP3_BRS) + " " + "c" * len(AAC_BRS)

    c0 = 5  # first Opus column
    c_opus = (c0, c0 + len(OPUS_BRS) - 1)
    c_mp3  = (c_opus[1] + 1, c_opus[1] + len(MP3_BRS))
    c_aac  = (c_mp3[1] + 1, c_mp3[1] + len(AAC_BRS))

    def hdr(brs):
        return " & ".join(f"\\textbf{{{b}k}}" for b in brs)

    rob_label = (
        f"\\multirow{{2}}{{*}}{{{args.scenario_label}}} & "
        f"\\multirow{{2}}{{*}}{{{args.model_label}}} & "
        f"Codec-robust EoT"
    )
    nrb_label = (
        f"                       &                       & "
        f"No EoT (clean train)"
    )

    rob_row = build_row(rob_label, rob, args.shade_threshold)
    nrb_row = build_row(nrb_label, nrb, args.shade_threshold)

    lines: list[str] = []
    lines.append("% Auto-generated by scripts/build_s3_eot_ablation_table.py")
    lines.append(f"% {args.scenario_label} / {args.model_label} / eps={args.eps}: codec-robust vs non-robust")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(
        r"\caption{\textbf{Codec-EoT ablation, "
        + f"{args.scenario_label} / {args.model_label} / "
        + f"$\\epsilon{{=}}{args.eps}$ "
        + (f"($n{{=}}{rob_n}$).}} "
           if rob_n == nrb_n else f"($n_{{\\text{{rob}}}}{{=}}{rob_n}$, $n_{{\\text{{nrb}}}}{{=}}{nrb_n}$).}} ")
        + r"Two attacks against the same target on identical carriers: "
        r"the headline \emph{codec-robust} attack (multi-bitrate Opus EoT during "
        r"training) versus the \emph{no-EoT} ablation (channel-mode=none). "
        r"Both reach near-perfect Clean ASR; the EoT step buys the entire "
        r"low-bitrate Opus tail (Opus $\le$32\,kbps drops to 0\% without EoT) "
        r"and recovers the AAC\,64k cell from 15\% to 47.5\%. "
        r"All MP3/AAC bitrates are held-out. "
        r"\colorbox{red!10}{Shaded}: $\geq$"
        + f"{args.shade_threshold:.0f}" + r"\% ASR.}"
    )
    lines.append(f"\\label{{tab:{args.scenario_label.lower()}_eot_ablation_eps{args.eps}}}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\toprule")
    lines.append(
        f" & & & & \\multicolumn{{{len(OPUS_BRS)}}}{{c}}{{\\textbf{{Opus}}}} "
        f"& \\multicolumn{{{len(MP3_BRS)}}}{{c}}{{\\textbf{{MP3 (held-out)}}}} "
        f"& \\multicolumn{{{len(AAC_BRS)}}}{{c}}{{\\textbf{{AAC-LC (held-out)}}}} \\\\"
    )
    lines.append(
        f"\\cmidrule(lr){{{c_opus[0]}-{c_opus[1]}}} "
        f"\\cmidrule(lr){{{c_mp3[0]}-{c_mp3[1]}}} "
        f"\\cmidrule(lr){{{c_aac[0]}-{c_aac[1]}}}"
    )
    lines.append(
        r"\textbf{Scenario} & \textbf{Model} & \textbf{Training} & \textbf{Clean} & "
        + hdr(OPUS_BRS) + " & " + hdr(MP3_BRS) + " & " + hdr(AAC_BRS) + r" \\"
    )
    lines.append(r"\midrule")
    lines.append(rob_row)
    lines.append(nrb_row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    out = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out)
    print(out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
