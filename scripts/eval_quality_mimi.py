"""
Audio quality eval for MIMI eot_full attacked wavs.

Stage 1: decode each of the 9 carriers through MIMI continuous round-trip
         (encoder -> decoder, skipping VQ) and save the result. This is the
         reference for SNR_delta (perturbation-only SNR).

Stage 2: for each (pair, channel) of 45 x 15 = 675, compute:
         SNR_carrier  = atk vs original carrier   (perceptual budget proxy)
         SNR_delta    = atk vs mimi(carrier)      (delta-only SNR)
         LSD_db       = log-spectral distance vs original carrier
         pesq_wb      = ITU-T P.862.2 WB at 16 kHz vs original carrier
         dLUFS        = LUFS(atk) - LUFS(carrier)

Outputs:
  quality/mimi_clean_roundtrip/<carrier>.wav
  quality/quality.jsonl
  quality/quality_rollup.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch, torchaudio

ROOT  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EVAL_ROLLUPS_DIR, MUSIC_DIR, RESULTS_DIR

# rollup.json ships with the repo; regenerated WAVs land under RESULTS_DIR.
# Point $BUNDLE_DIR elsewhere to score a different run.
EOT   = Path(os.environ.get(
    "BUNDLE_DIR",
    Path(EVAL_ROLLUPS_DIR) / "03b_copyright_bypass/mimi_eps0.2",
))
QDIR  = Path(RESULTS_DIR) / "03b_copyright_bypass/mimi_eps0.2/quality"
RTDIR = QDIR / "mimi_clean_roundtrip"
CARS  = Path(MUSIC_DIR) / "copyrighted"

QDIR.mkdir(parents=True, exist_ok=True)
RTDIR.mkdir(parents=True, exist_ok=True)

ROLL = json.load(open(EOT / "rollup.json"))
PAIRS = ROLL["pairs"]
CHANS = ROLL["channels"]
PAIR_CARRIER = {c["pair"]: c["carrier"] for c in ROLL["cells"] if c["channel"] == "clean"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 24000


def load24k(path):
    w, sr = sf.read(str(path), always_2d=False)
    if w.ndim > 1:
        w = w.mean(-1)
    w = torch.from_numpy(w.astype(np.float32))
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w.numpy()


def stage1_roundtrip():
    from mimi_wrapper import MimiCodecWrapper

    print("[stage1] loading MIMI codec...")
    codec = MimiCodecWrapper(device=DEVICE)
    carriers = sorted({c.stem: c for c in CARS.glob("*.mp3")}.items())

    for name, path in carriers:
        out = RTDIR / f"{name}.wav"
        if out.exists():
            print(f"  skip {name} (exists)")
            continue
        wav = load24k(path)
        x = torch.from_numpy(wav).to(DEVICE).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            z = codec.encode_to_continuous(x)
            y = codec.decode_from_continuous(z).squeeze().cpu().numpy()
        n = min(len(wav), len(y))
        sf.write(str(out), y[:n].astype(np.float32), SR, subtype="FLOAT")
        print(f"  wrote {out.name}  in_samples={len(wav)}  out_samples={len(y)}")
    del codec
    torch.cuda.empty_cache()


def snr_db(sig, noise):
    p_s = float(np.sum(sig.astype(np.float64) ** 2))
    p_n = float(np.sum(noise.astype(np.float64) ** 2))
    if p_n <= 0:
        return float("inf")
    return 10.0 * np.log10(p_s / p_n)


def lsd_db(ref, deg, n_fft=1024, hop=256):
    R = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(ref, n_fft)[::hop], axis=-1))
    D = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(deg, n_fft)[::hop], axis=-1))
    eps = 1e-8
    return float(np.mean(np.sqrt(np.mean((10*np.log10(R + eps) - 10*np.log10(D + eps))**2, axis=-1))))


def channel_routed_path(pair_idx, carrier, channel):
    if channel == "clean":
        return EOT / "channels/clean" / f"music_{carrier}__pair{pair_idx:03d}_attacked__clean.wav"
    return EOT / f"channels/{channel}" / f"music_{carrier}__pair{pair_idx:03d}_attacked__{channel}.wav"


def stage2_compute():
    from pesq import pesq, PesqError
    import pyloudnorm as pyln

    meter = pyln.Meter(SR)

    carrier_cache = {}
    for name in set(PAIR_CARRIER.values()):
        x_car = load24k(CARS / f"{name}.mp3")
        x_rt  = load24k(RTDIR / f"{name}.wav")
        n = min(len(x_car), len(x_rt))
        x_car, x_rt = x_car[:n], x_rt[:n]
        car_lufs = float(meter.integrated_loudness(x_car))
        car_16k = torchaudio.functional.resample(torch.from_numpy(x_car), SR, 16000).numpy()
        carrier_cache[name] = dict(x_car=x_car, x_rt=x_rt, lufs=car_lufs, x_car_16k=car_16k)

    out_path = QDIR / "quality.jsonl"
    out = open(out_path, "w")
    t0 = time.time()
    n_cells = 0
    n_total = len(PAIRS) * len(CHANS)

    for pair_idx in PAIRS:
        carrier = PAIR_CARRIER[pair_idx]
        c = carrier_cache[carrier]

        for channel in CHANS:
            path = channel_routed_path(pair_idx, carrier, channel)
            if not path.is_file():
                row = dict(pair=pair_idx, carrier=carrier, channel=channel,
                           error="missing", path=str(path))
                out.write(json.dumps(row) + "\n")
                continue
            atk = load24k(path)
            n = min(len(atk), len(c["x_car"]))
            atk, x_car, x_rt = atk[:n], c["x_car"][:n], c["x_rt"][:n]

            snr_car = snr_db(x_car, atk - x_car)
            snr_del = snr_db(x_rt,  atk - x_rt)
            lsd = lsd_db(x_car, atk)

            atk_16k = torchaudio.functional.resample(torch.from_numpy(atk), SR, 16000).numpy()
            try:
                p = float(pesq(16000, c["x_car_16k"][:len(atk_16k)], atk_16k, "wb"))
            except (PesqError, Exception):
                p = None

            try:
                atk_lufs = float(meter.integrated_loudness(atk))
                dlufs = atk_lufs - c["lufs"]
            except Exception:
                atk_lufs = None; dlufs = None

            row = dict(
                pair=pair_idx, carrier=carrier, channel=channel,
                snr_carrier_db=round(snr_car, 3),
                snr_delta_db=round(snr_del, 3),
                lsd_db=round(lsd, 3),
                pesq_wb=None if p is None else round(p, 3),
                lufs_atk=None if atk_lufs is None else round(atk_lufs, 3),
                lufs_carrier=round(c["lufs"], 3),
                dlufs=None if dlufs is None else round(dlufs, 3),
            )
            out.write(json.dumps(row) + "\n")
            n_cells += 1
            if n_cells % 50 == 0:
                print(f"  [{n_cells}/{n_total}]  {time.time()-t0:.0f}s")
    out.close()
    print(f"[stage2] wrote {out_path}  cells={n_cells}  in {time.time()-t0:.0f}s")


def stage3_rollup():
    rows = [json.loads(l) for l in open(QDIR / "quality.jsonl")]
    by_ch = {}
    for r in rows:
        if r.get("error"):
            continue
        by_ch.setdefault(r["channel"], []).append(r)

    def stats(values):
        v = np.array([x for x in values if x is not None and not np.isinf(x)])
        if len(v) == 0:
            return None
        return dict(n=int(len(v)),
                    mean=round(float(v.mean()), 3),
                    median=round(float(np.median(v)), 3),
                    p10=round(float(np.percentile(v, 10)), 3),
                    p90=round(float(np.percentile(v, 90)), 3))

    summary = {}
    for ch in CHANS:
        if ch not in by_ch:
            summary[ch] = None
            continue
        rs = by_ch[ch]
        summary[ch] = dict(
            snr_carrier_db = stats([r["snr_carrier_db"] for r in rs]),
            snr_delta_db   = stats([r["snr_delta_db"]   for r in rs]),
            lsd_db         = stats([r["lsd_db"]         for r in rs]),
            pesq_wb        = stats([r["pesq_wb"]        for r in rs]),
            dlufs          = stats([r["dlufs"]          for r in rs]),
            n_cells = len(rs),
        )
    out = QDIR / "quality_rollup.json"
    json.dump(dict(eps_mimi=ROLL["eps_mimi"], steps=ROLL["steps"],
                   pairs=len(PAIRS), channels=len(CHANS),
                   by_channel=summary), open(out, "w"), indent=2)
    print(f"[stage3] wrote {out}")


if __name__ == "__main__":
    stage1_roundtrip()
    stage2_compute()
    stage3_rollup()
