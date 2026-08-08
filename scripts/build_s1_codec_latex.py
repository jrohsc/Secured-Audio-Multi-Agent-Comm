"""
Emit a standalone booktabs LaTeX table comparing latent vs waveform S1
attacks on Qwen2-Audio across Opus / MP3 / AAC bitrates.

Reuses the same aggregation logic as compare_s1_bundleA_latent_vs_waveform.py
and writes to paper/tables/s1_latent_vs_waveform_codecs.tex.

Usage:
    python scripts/build_s1_codec_latex.py
"""
from __future__ import annotations

import argparse
import json
import glob
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
CODEC_ROOT = ROOT
SCEN = "01_finance_voice_agent"
DEFAULT_LATENT_DIR = ROLLUPS / f"{SCEN}/qwen2_audio/eps_1.0/audio"
DEFAULT_WAVE_ROOT = ROLLUPS / f"_results_waveform/{SCEN}/qwen2_audio"
DEFAULT_OUT = TABLES_OUT / "tab_latent_vs_waveform.tex"

Z_95 = 1.959963984540054  # exact 1 - 0.025 quantile of standard normal

OPUS_BITRATES = ["16", "24", "32", "64", "128", "192"]
MP3_BITRATES  = ["64", "96", "128", "192"]
AAC_BITRATES  = ["64", "96", "128", "192"]
CODEC_SPECS = [
    ("opus", "Opus", OPUS_BITRATES),
    ("mp3",  "MP3",  MP3_BITRATES),
    ("aac",  "AAC",  AAC_BITRATES),
]


def _carrier_stem(clip: str) -> str | None:
    m = re.match(r"^music_(.+?)__target_", clip)
    return m.group(1) if m else None


def _load_json_map(glob_path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(glob_path)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        stem = _carrier_stem(d.get("clip", ""))
        if stem:
            out[stem] = d
    return out


def _pick_wave_eps_dir(wave_root: Path, eps_arg: float | None) -> Path:
    if eps_arg is not None:
        cand = wave_root / f"eps_{eps_arg:g}"
        if not cand.is_dir():
            raise SystemExit(f"no waveform bundle at {cand}")
        return cand / "audio"
    eps_dirs = sorted(p for p in wave_root.iterdir() if p.is_dir() and p.name.startswith("eps_"))
    if not eps_dirs:
        raise SystemExit(f"no eps_* subdirs under {wave_root}")
    target = next((p for p in eps_dirs if p.name == "eps_0.075"), eps_dirs[-1])
    return target / "audio"


def _rate(summaries: dict[str, dict], codecs: dict[str, dict],
          carriers: list[str], codec: str, bw: str) -> tuple[int, int]:
    """Return (hits, n) using STRICT substring match of target in raw output.

    Mirrors the metric used by build_main_results_tables.py:184 so all paper
    tables agree. Reads the per-codec `output` field and recomputes
    `target_text.lower() in output.lower()` rather than trusting the cached
    `matches_target` flag (which uses a more lenient fuzzy criterion and was
    the source of a metric inconsistency between this table and tab_S1).
    """
    key = f"{codec}_robustness"
    src = summaries if codec == "opus" else codecs
    hits = 0
    n = 0
    for c in carriers:
        d = src.get(c)
        if d is None:
            continue
        row = (d.get(key) or {}).get(bw)
        if not isinstance(row, dict):
            continue
        target = (d.get("target_text") or "").strip().lower()
        out = (row.get("output") or "").strip().lower()
        if not target:
            continue
        n += 1
        if target in out:
            hits += 1
    return (hits, n)


def _wilson_half_pp(k: int, n: int) -> float:
    """Wilson 95% CI half-width in percentage points. 0 if n==0."""
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1.0 + Z_95 * Z_95 / n
    half = Z_95 * math.sqrt(p * (1.0 - p) / n + Z_95 * Z_95 / (4.0 * n * n)) / denom
    return 100.0 * half


def _fmt_cell(k: int, n: int) -> str:
    """Render $p_{\\pm h}$ matching tab_S1 / tab_S2 / tab_S3 format."""
    if n <= 0:
        return r"\pending"
    pct = 100.0 * k / n
    half = _wilson_half_pp(k, n)
    return f"${pct:5.1f}_{{\\pm {half:.0f}}}$"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT_DIR)
    p.add_argument("--wave-root",  type=Path, default=DEFAULT_WAVE_ROOT)
    p.add_argument("--wave-eps",   type=float, default=None)
    p.add_argument("--out",        type=Path, default=DEFAULT_OUT)
    p.add_argument("--latent-snr-db", type=float, default=5.77)
    p.add_argument("--wave-snr-db",   type=float, default=5.8)
    args = p.parse_args()

    wave_audio_dir = _pick_wave_eps_dir(args.wave_root, args.wave_eps)
    wave_eps_tag = wave_audio_dir.parent.name  # "eps_0.075"

    lat_summ = _load_json_map(str(args.latent_dir / "*_summary.json"))
    lat_codec = _load_json_map(str(args.latent_dir / "*_codecs.json"))
    wav_summ = _load_json_map(str(wave_audio_dir / "*_summary.json"))
    wav_codec = _load_json_map(str(wave_audio_dir / "*_codecs.json"))
    common = sorted(set(lat_summ.keys()) & set(wav_summ.keys()))
    n_common = len(common)
    if not common:
        raise SystemExit("no common carriers")

    # Build per-codec per-bitrate (hits, n) tuples for both attacks
    rows = {}
    for codec, _lbl, bitrates in CODEC_SPECS:
        for bw in bitrates:
            lat = _rate(lat_summ, lat_codec, common, codec, bw)
            wav = _rate(wav_summ, wav_codec, common, codec, bw)
            rows[(codec, bw)] = (lat, wav)

    # Column count: 2 (Attack, n) + 6 (Opus) + 5 (MP3) + 5 (AAC) = 18
    n_opus, n_mp3, n_aac = len(OPUS_BITRATES), len(MP3_BITRATES), len(AAC_BITRATES)

    # Column spec: first two left-aligned, rest centered
    colspec = "ll" + "c" * (n_opus + n_mp3 + n_aac)

    # Multicolumn header ranges (1-indexed columns)
    c0 = 3
    c_opus = (c0, c0 + n_opus - 1)
    c_mp3  = (c_opus[1] + 1, c_opus[1] + n_mp3)
    c_aac  = (c_mp3[1] + 1, c_mp3[1] + n_aac)

    def _br_header(bitrates):
        return " & ".join(f"\\textbf{{{bw}k}}" for bw in bitrates)

    lines = []
    lines.append(f"% Auto-generated by scripts/build_s1_codec_latex.py")
    lines.append(f"% Scenario: {SCEN}, Qwen2-Audio, latent eps=1.0 vs waveform {wave_eps_tag},")
    lines.append(f"% n={n_common} common S1 carriers.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(
        r"\caption{S1 finance voice-agent: latent-space vs.\ waveform-space attack "
        r"on Qwen2-Audio at matched clean SNR "
        f"(latent $\\epsilon{{=}}1.0$, {args.latent_snr_db:.2f}\\,dB; waveform "
        f"$\\epsilon{{=}}0.075$, $\\approx${args.wave_snr_db:.2f}\\,dB). "
        r"Target-substring match (\%) after the adversarial audio is transcoded "
        r"through each lossy codec at the listed bitrate, then re-fed to the "
        r"target model. Both attacks are trained with multi-bitrate \emph{Opus} "
        r"EoT only; MP3 and AAC are held-out codecs (inference-only eval, no "
        r"retraining). "
        r"\textbf{Headline:} latent-space attack beats waveform on every "
        r"in-distribution Opus bitrate and generalizes to MP3 out-of-the-box, "
        r"but both attacks collapse on AAC's psychoacoustic masking.}"
    )
    lines.append(r"\label{tab:s1_latent_vs_waveform_codecs}")
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")
    lines.append(
        f" & & \\multicolumn{{{n_opus}}}{{c}}{{\\textbf{{Opus (in-distribution)}}}} "
        f"& \\multicolumn{{{n_mp3}}}{{c}}{{\\textbf{{MP3 (held-out)}}}} "
        f"& \\multicolumn{{{n_aac}}}{{c}}{{\\textbf{{AAC (held-out)}}}} \\\\"
    )
    lines.append(
        f"\\cmidrule(lr){{{c_opus[0]}-{c_opus[1]}}} "
        f"\\cmidrule(lr){{{c_mp3[0]}-{c_mp3[1]}}} "
        f"\\cmidrule(lr){{{c_aac[0]}-{c_aac[1]}}}"
    )
    lines.append(
        r"\textbf{Attack} & \textbf{$n$} & "
        + _br_header(OPUS_BITRATES) + " & "
        + _br_header(MP3_BITRATES)  + " & "
        + _br_header(AAC_BITRATES)  + r" \\"
    )
    lines.append(r"\midrule")

    # Latent row
    cells = []
    for codec, _lbl, bitrates in CODEC_SPECS:
        for bw in bitrates:
            (lk, ln), _wav = rows[(codec, bw)]
            cells.append(_fmt_cell(lk, ln))
    lines.append(
        r"Latent ($\epsilon{=}1.0$) & " + f"{n_common} & "
        + " & ".join(cells) + r" \\"
    )

    # Waveform row
    cells = []
    for codec, _lbl, bitrates in CODEC_SPECS:
        for bw in bitrates:
            _lat, (wk, wn) = rows[(codec, bw)]
            cells.append(_fmt_cell(wk, wn))
    lines.append(
        r"Waveform ($\epsilon{=}0.075$) & " + f"{n_common} & "
        + " & ".join(cells) + r" \\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out)
    print(out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
