"""
Audio quality eval for EnCodec eps=1.0 S3b clean attacked wavs (n=45).

Mirrors eval_quality_mimi.py but for EnCodec. Channel-routed wavs are
not available for this bundle, so quality is reported on the clean
attacked wav only (which is the right reference for perceptual budget).

Outputs:
  quality/encodec_clean_roundtrip/<carrier>.wav
  quality/quality.jsonl
  quality/quality_rollup.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch, torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EVAL_ROLLUPS_DIR, MUSIC_DIR, RESULTS_DIR

# results.jsonl ships with the repo; regenerated WAVs land under RESULTS_DIR.
# Point $BUNDLE_DIR elsewhere to score a different run.
EN   = Path(os.environ.get(
    "BUNDLE_DIR",
    Path(EVAL_ROLLUPS_DIR) / "03b_copyright_bypass/encodec_eps1.0",
))
QDIR = Path(RESULTS_DIR) / "03b_copyright_bypass/encodec_eps1.0/quality"
RTDIR = QDIR / "encodec_clean_roundtrip"
CARS = Path(MUSIC_DIR) / "copyrighted"

QDIR.mkdir(parents=True, exist_ok=True)
RTDIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 24000

ROWS = [json.loads(l) for l in open(EN / "results.jsonl")]
PAIR_ENTRIES = []
for r in ROWS:
    p = r["adv_audio_path"]
    pair_idx = int(p.split("_pair")[-1].split("_")[0])
    carrier = r["carrier"].replace(".mp3", "")
    candidates = [EN / p, ROOT / p, Path(RESULTS_DIR) / p]
    atk_path = next((c for c in candidates if c.is_file()), candidates[0])
    PAIR_ENTRIES.append(dict(pair=pair_idx, carrier=carrier, atk_path=atk_path,
                             target=r["target_command"]))


def load24k(path):
    w, sr = sf.read(str(path), always_2d=False)
    if w.ndim > 1:
        w = w.mean(-1)
    w = torch.from_numpy(w.astype(np.float32))
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w.numpy()


def stage1_roundtrip():
    from encodec_wrapper import EnCodecWrapper

    print("[stage1] loading EnCodec codec...")
    codec = EnCodecWrapper(bandwidth=6.0, device=DEVICE)
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


def stage2_compute():
    from pesq import pesq, PesqError
    import pyloudnorm as pyln

    meter = pyln.Meter(SR)

    carrier_cache = {}
    for entry in PAIR_ENTRIES:
        name = entry["carrier"]
        if name in carrier_cache:
            continue
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

    for entry in PAIR_ENTRIES:
        carrier = entry["carrier"]
        c = carrier_cache[carrier]
        path = entry["atk_path"]
        if not path.is_file():
            row = dict(pair=entry["pair"], carrier=carrier, channel="clean",
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
            pair=entry["pair"], carrier=carrier, channel="clean",
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
    out.close()
    print(f"[stage2] wrote {out_path}  cells={n_cells}  in {time.time()-t0:.0f}s")


def stage3_rollup():
    rows = [json.loads(l) for l in open(QDIR / "quality.jsonl")]
    rs = [r for r in rows if not r.get("error")]

    def stats(values):
        v = np.array([x for x in values if x is not None and not np.isinf(x)])
        if len(v) == 0:
            return None
        return dict(n=int(len(v)),
                    mean=round(float(v.mean()), 3),
                    median=round(float(np.median(v)), 3),
                    p10=round(float(np.percentile(v, 10)), 3),
                    p90=round(float(np.percentile(v, 90)), 3))

    summary = dict(
        snr_carrier_db = stats([r["snr_carrier_db"] for r in rs]),
        snr_delta_db   = stats([r["snr_delta_db"]   for r in rs]),
        lsd_db         = stats([r["lsd_db"]         for r in rs]),
        pesq_wb        = stats([r["pesq_wb"]        for r in rs]),
        dlufs          = stats([r["dlufs"]          for r in rs]),
        n_cells = len(rs),
    )
    out = QDIR / "quality_rollup.json"
    json.dump(dict(eps_encodec=1.0, codec="encodec_24khz", channel="clean",
                   pairs=len(rs), by_channel={"clean": summary}), open(out, "w"), indent=2)
    print(f"[stage3] wrote {out}")


if __name__ == "__main__":
    stage1_roundtrip()
    stage2_compute()
    stage3_rollup()
