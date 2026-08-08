"""Aggregate S1 latent-vs-waveform exact-match ASR + Wilson 95% CI per
codec/bitrate, and emit the LaTeX rows for tab_latent_vs_waveform.tex.

Usage:
    python scripts/build_s1_latent_vs_waveform_table.py
        [--latent-dir results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0]
        [--waveform-dir results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EVAL_ROLLUPS_DIR, PROJECT_ROOT

ROOT = Path(PROJECT_ROOT)
# Reference rollups ship in data/eval_rollups/; generated .tex goes to results/tables/.
ROLLUPS = Path(EVAL_ROLLUPS_DIR)
TABLES_OUT = ROOT / "results" / "tables"

OPUS_BITRATES = [16, 24, 32, 64, 96, 128, 192]
MP3_BITRATES  = [16, 24, 32, 64, 96, 128, 192]
AAC_BITRATES  = [16, 24, 32, 64, 96, 128, 192]


def _bundle_records(bundle_dir: Path):
    """Yield (summary_dict, codecs_dict_or_None) per clip in audio/."""
    audio = bundle_dir / "audio"
    for sumf in sorted(audio.glob("*_summary.json")):
        cdcf = sumf.with_name(sumf.stem.replace("_summary", "") + "_codecs.json")
        sd = json.load(open(sumf))
        cd = json.load(open(cdcf)) if cdcf.is_file() else None
        yield sd, cd


def _exact_match_count(records, key: str, bw: int):
    """Count exact_match across clips. For opus_robustness, the training
    runner writes a fixed grid into _summary.json; held-out bitrates added
    post-hoc by eval_opus_bundle.py live in _codecs.json. Check both."""
    hits = 0
    n = 0
    for sd, cd in records:
        cell = None
        if key == "opus_robustness":
            cell = ((sd.get(key) or {}).get(str(bw))
                    or ((cd or {}).get(key) or {}).get(str(bw)))
        else:
            cell = ((cd or {}).get(key) or {}).get(str(bw))
        if cell is None:
            continue
        n += 1
        if cell.get("exact_match"):
            hits += 1
    return hits, n


def wilson_halfwidth(hits: int, n: int, z: float = 1.96) -> float:
    """Wilson 95% CI half-width (returns half of upper-lower spread, in [0,1])."""
    if n == 0:
        return 0.0
    p = hits / n
    denom = 1.0 + (z * z) / n
    margin = z * math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))
    half = margin / denom
    return half


def fmt_cell(hits: int, n: int) -> str:
    p = (hits / n * 100.0) if n else 0.0
    hw = wilson_halfwidth(hits, n) * 100.0
    return f"${p:.1f}_{{\\pm {round(hw)}}}$"


def fmt_pair(lat: tuple, wav: tuple) -> tuple[str, str]:
    """Return (latent_cell, waveform_cell) with red shading on the strictly
    higher exact-match rate. Ties → no shading (matches existing tex when
    both are 0/0)."""
    lh, ln = lat
    wh, wn = wav
    lc = fmt_cell(lh, ln)
    wc = fmt_cell(wh, wn)
    lp = (lh / ln) if ln else 0.0
    wp = (wh / wn) if wn else 0.0
    if ln == 0 and wn == 0:
        return "--", "--"
    if ln == 0:
        return "--", wc
    if wn == 0:
        return lc, "--"
    if lp > wp:
        lc = r"\cellcolor{red!10}" + lc
    elif wp > lp:
        wc = r"\cellcolor{red!10}" + wc
    return lc, wc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent-dir", type=Path,
                   default=ROOT / "results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0")
    p.add_argument("--waveform-dir", type=Path,
                   default=ROOT / "results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075")
    args = p.parse_args()

    lat_recs = list(_bundle_records(args.latent_dir))
    wav_recs = list(_bundle_records(args.waveform_dir))
    print(f"% latent: {len(lat_recs)} clips  | waveform: {len(wav_recs)} clips")

    bitrates = sorted(set(OPUS_BITRATES) | set(MP3_BITRATES) | set(AAC_BITRATES))
    rows = []
    for bw in bitrates:
        opus_l = _exact_match_count(lat_recs, "opus_robustness", bw)
        opus_w = _exact_match_count(wav_recs, "opus_robustness", bw)
        mp3_l  = _exact_match_count(lat_recs, "mp3_robustness", bw)
        mp3_w  = _exact_match_count(wav_recs, "mp3_robustness", bw)
        aac_l  = _exact_match_count(lat_recs, "aac_robustness", bw)
        aac_w  = _exact_match_count(wav_recs, "aac_robustness", bw)

        olc, owc = fmt_pair(opus_l, opus_w)
        mlc, mwc = fmt_pair(mp3_l,  mp3_w)
        alc, awc = fmt_pair(aac_l,  aac_w)

        # Skip a row only if every codec is empty for both attacks
        if all(x == ("--", "--") for x in [(olc, owc), (mlc, mwc), (alc, awc)]):
            continue

        rows.append((bw, olc, owc, mlc, mwc, alc, awc,
                     opus_l, opus_w, mp3_l, mp3_w, aac_l, aac_w))

    # Print LaTeX-ready rows
    print()
    print("% --- copy into tab_latent_vs_waveform.tex (between \\midrule and \\bottomrule) ---")
    for bw, olc, owc, mlc, mwc, alc, awc, *_ in rows:
        bw_str = f"{bw}\\,kbps"
        print(f"{bw_str:<10} & {olc} & {owc} & {mlc} & {mwc} & {alc} & {awc} \\\\")

    # Also print the raw counts so it's auditable
    print()
    print("% --- raw exact-match counts (hits / n) ---")
    print("% bitrate | opus_lat | opus_wav | mp3_lat | mp3_wav | aac_lat | aac_wav")
    for bw, *_, opus_l, opus_w, mp3_l, mp3_w, aac_l, aac_w in rows:
        def fmt(t): return f"{t[0]:>2}/{t[1]:<2}"
        print(f"% {bw:>4} | {fmt(opus_l)} | {fmt(opus_w)} | {fmt(mp3_l)} | {fmt(mp3_w)} | {fmt(aac_l)} | {fmt(aac_w)}")


if __name__ == "__main__":
    main()
