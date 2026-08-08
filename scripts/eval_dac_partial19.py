"""
Route + score the 19 DAC eot_full attacked wavs that already exist on disk
across the full 15-channel grid (1 clean + 6 Opus + 4 MP3 + 4 AAC).

No attack is launched; this only consumes the existing audio/*.wav files.
Writes channels/<ch>/*.wav and rollup_n19.json next to the existing rollup.json.
"""
import os, sys, json, time, traceback
from pathlib import Path

# Pin TMPDIR before any tempfile call.
os.environ.setdefault("TMPDIR",
    os.path.join(os.path.expanduser("~"), ".cache", "codec_attack_scratch"))
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

# Reuse helpers from run_dac_full (which also sets up sys.path via config).
from run_dac_full import (
    CHANNELS_FULL, DEVICE, MODEL_PATH, lookup_pair,
    channel_route_and_save, eval_wav,
    EPS_DAC, ATTACK_STEPS, EOT_OPUS_BITRATES,
)
import torch
from config import RESULTS_DIR
from models.qwen25_omni import Qwen25OmniModel

OUT = Path(RESULTS_DIR) / (
    "03b_copyright_bypass/dac_eps0.6194/eot_full"
)
AUDIO = OUT / "audio"

print(f"[init] audio dir: {AUDIO}")
attacked = sorted(AUDIO.glob("*.wav"))
print(f"[init] found {len(attacked)} attacked wavs")

pair_meta = []
for w in attacked:
    name = w.name
    # filename: music_<carrier>__pair<NNN>_attacked.wav
    pair_idx = int(name.split("_pair")[1].split("_")[0])
    summary = lookup_pair(pair_idx)
    carrier_stem = Path(summary["wav_path"]).stem
    pair_meta.append({
        "pair_idx": pair_idx,
        "carrier_stem": carrier_stem,
        "target_text": summary["target_text"],
        "attacked_wav": str(w),
    })
pair_meta.sort(key=lambda m: m["pair_idx"])
print(f"[init] resolved {len(pair_meta)} pairs: {[m['pair_idx'] for m in pair_meta]}")

print(f"[init] loading Qwen2.5-Omni on {DEVICE}...")
model = Qwen25OmniModel(model_path=MODEL_PATH, device=DEVICE, dtype=torch.bfloat16)

# ---- Phase 2: routing ----
print(f"\n{'-'*70}\n[routing] {len(pair_meta)} pairs x {len(CHANNELS_FULL)} channels")
routed = {}
t0 = time.time()
for ch_tag, ch_kind, ch_param in CHANNELS_FULL:
    ch_dir = OUT / "channels" / ch_tag
    for meta in pair_meta:
        out_path = ch_dir / (Path(meta["attacked_wav"]).stem + f"__{ch_tag}.wav")
        if out_path.is_file():
            routed[(meta["pair_idx"], ch_tag)] = out_path
            continue
        try:
            p = channel_route_and_save(Path(meta["attacked_wav"]),
                                       ch_tag, ch_kind, ch_param, ch_dir)
            routed[(meta["pair_idx"], ch_tag)] = p
        except Exception as e:
            print(f"  ERR routing pair{meta['pair_idx']:03d} {ch_tag}: {e}", flush=True)
    n = sum(1 for k in routed if k[1] == ch_tag)
    print(f"  [{ch_tag}] routed {n}/{len(pair_meta)}", flush=True)
print(f"[routing] elapsed {time.time()-t0:.0f}s")

# ---- Phase 3: eval ----
print(f"\n{'-'*70}\n[eval] scoring {len(routed)} cells")
cells = []
t0 = time.time()
for meta in pair_meta:
    for ch_tag, _, _ in CHANNELS_FULL:
        key = (meta["pair_idx"], ch_tag)
        if key not in routed:
            continue
        try:
            r = eval_wav(model, routed[key], meta["target_text"])
        except Exception as e:
            r = {"success": False, "wer": 1.0, "output": f"ERROR: {e}"}
            traceback.print_exc()
        flag = "PASS" if r["success"] else "FAIL"
        print(f"  [{flag}] pair{meta['pair_idx']:03d} {ch_tag:9s} "
              f"wer={r['wer']:.2f} out={r['output'][:70]!r}", flush=True)
        cells.append({"pair": meta["pair_idx"], "carrier": meta["carrier_stem"],
                      "channel": ch_tag, "target_text": meta["target_text"], **r})
print(f"[eval] elapsed {time.time()-t0:.0f}s")

# ---- Phase 4: rollup ----
by_ch = {}
for ch_tag, _, _ in CHANNELS_FULL:
    cs = [c for c in cells if c["channel"] == ch_tag]
    n_succ = sum(1 for c in cs if c["success"])
    by_ch[ch_tag] = {"n": len(cs), "n_success": n_succ,
                     "asr_pct": round(100*n_succ/len(cs), 1) if cs else 0.0}

rollup = {
    "mode": "partial_n19",
    "pairs": [m["pair_idx"] for m in pair_meta],
    "channels": [c[0] for c in CHANNELS_FULL],
    "eps_dac": EPS_DAC,
    "steps": ATTACK_STEPS,
    "eot_opus_bitrates": EOT_OPUS_BITRATES,
    "cells": cells,
    "summary": {"by_channel": by_ch},
}
out_path = OUT / "rollup_n19.json"
out_path.write_text(json.dumps(rollup, indent=2))
print(f"\n[done] wrote {out_path}")

print("\n## DAC EoT (n=19, partial) vs EnCodec eps=1.0 (n=45 ref)")
enc_ref = {"clean":100.0, "opus192k":100.0, "opus128k":100.0, "opus64k":93.3,
           "opus32k":64.4, "opus24k":46.7, "opus16k":28.9}
print(f"  {'channel':<10} {'DAC':<14} {'EnCodec ref':<14}")
for ch_tag, _, _ in CHANNELS_FULL:
    d = by_ch[ch_tag]
    ref = enc_ref.get(ch_tag, "-")
    print(f"  {ch_tag:<10} {d['n_success']}/{d['n']} ({d['asr_pct']:5.1f}%)  {ref}")
