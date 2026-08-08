"""
Over-the-Air (OTA) adversarial audio demo.

Simulates the YouTube/Spotify playback scenario:
  Adversarial WAV → Platform compression (Opus/AAC) → Speaker → Air → Microphone → Audio LLM

Channel simulation layers (applied sequentially):
  1. Opus compression (YouTube/Spotify re-encode at ~128-256 kbps)
  2. Bandpass filter (speaker + microphone frequency response, ~100 Hz – 14 kHz)
  3. Additive noise (ambient noise, SNR 20–50 dB)
  4. Volume scaling (distance variation)

Supports two modes:
  --simulate : Apply synthetic channel degradation to existing adversarial audio
  --record   : Record from microphone (for real OTA demo with physical speaker)

Usage:
    # Simulate OTA on a single file
    python demo_ota.py --simulate --audio results_transcribe/.../audio/jazz_1__nav_turn_left.wav \
                       --target "Turn left at the next intersection" --model qwen2_audio

    # Simulate OTA on an entire benchmark directory
    python demo_ota.py --simulate --benchmark results_transcribe/benchmark_qwen2_audio_eps0.5_jazz_1 \
                       --model qwen2_audio --snr 30

    # Record from microphone (real OTA)
    python demo_ota.py --record --duration 10 --model qwen2_audio

    # Play adversarial audio through speaker, then record & evaluate
    python demo_ota.py --play-and-record --audio results_transcribe/.../audio/jazz_1__nav_turn_left.wav \
                       --target "Turn left at the next intersection" --model qwen2_audio
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CODECATTACK_LIB_ROOT, MODEL_PATHS,
    ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    PROMPT_MODES, DEFAULT_PROMPT_MODE,
)
sys.path.insert(0, CODECATTACK_LIB_ROOT)


# ============================================================================
# OTA Channel Simulation
# ============================================================================

def _make_tempdir_resilient() -> str:
    # Some shared-cluster nodes have a flaky /tmp where mkdtemp races against a
    # cleanup process and raises FileNotFoundError. Retry a few times, then fall
    # back to a stable per-user scratch dir under $HOME.
    import shutil as _shutil  # local to avoid touching module-level imports
    for attempt in range(4):
        try:
            return tempfile.mkdtemp()
        except FileNotFoundError:
            time.sleep(0.2 * (2 ** attempt))
    home_scratch = os.path.join(os.path.expanduser("~"), ".cache", "codec_attack_scratch")
    os.makedirs(home_scratch, exist_ok=True)
    return tempfile.mkdtemp(dir=home_scratch)


def apply_opus_compression(audio_np: np.ndarray, sr: int, bitrate_kbps: int = 128) -> np.ndarray:
    """Compress audio through Opus codec (simulates YouTube/Spotify encoding)."""
    import shutil

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        # Try conda env
        env_bin = os.path.dirname(sys.executable)
        ffmpeg_bin = os.path.join(env_bin, "ffmpeg")
        if not os.path.isfile(ffmpeg_bin):
            print("  [WARN] ffmpeg not found, skipping Opus compression")
            return audio_np

    tmpdir = _make_tempdir_resilient()
    try:
        in_path = os.path.join(tmpdir, "input.wav")
        opus_path = os.path.join(tmpdir, "compressed.opus")
        out_path = os.path.join(tmpdir, "output.wav")

        sf.write(in_path, audio_np, sr)

        subprocess.run(
            [ffmpeg_bin, "-y", "-i", in_path, "-c:a", "opus", "-strict", "-2",
             "-b:a", f"{bitrate_kbps}k", opus_path],
            capture_output=True, check=True,
        )
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", opus_path, "-ar", str(sr), out_path],
            capture_output=True, check=True,
        )

        decoded_np, _ = sf.read(out_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return decoded_np


def apply_bandpass_filter(audio_np: np.ndarray, sr: int,
                          low_hz: float = 100.0, high_hz: float = 14000.0) -> np.ndarray:
    """Bandpass filter simulating speaker + microphone frequency response."""
    from scipy.signal import butter, sosfilt

    nyquist = sr / 2.0
    low = max(low_hz / nyquist, 0.001)
    high = min(high_hz / nyquist, 0.999)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfilt(sos, audio_np).astype(np.float32)


def apply_additive_noise(audio_np: np.ndarray, snr_db: float = 30.0) -> np.ndarray:
    """Add white Gaussian noise at the given SNR."""
    signal_power = np.mean(audio_np ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*audio_np.shape).astype(np.float32) * np.sqrt(noise_power)
    return (audio_np + noise).astype(np.float32)


def apply_volume_scaling(audio_np: np.ndarray, gain_db: float = 0.0) -> np.ndarray:
    """Apply volume scaling (simulates distance variation)."""
    gain = 10 ** (gain_db / 20)
    return (audio_np * gain).astype(np.float32)


def simulate_ota_channel(
    audio_np: np.ndarray,
    sr: int,
    opus_bitrate: int = 128,
    snr_db: float = 30.0,
    bandpass: bool = True,
    low_hz: float = 100.0,
    high_hz: float = 14000.0,
    gain_db: float = -3.0,
) -> np.ndarray:
    """
    Apply full OTA channel simulation (synthetic).

    Pipeline:
        1. Opus compression (platform re-encoding)
        2. Bandpass filter (speaker + mic)
        3. Additive noise (ambient)
        4. Volume scaling (distance)
    """
    audio = audio_np.copy()

    # 1. Platform compression
    if opus_bitrate > 0:
        orig_len = len(audio)
        audio = apply_opus_compression(audio, sr, opus_bitrate)
        # Match lengths (Opus can add/remove samples)
        if len(audio) > orig_len:
            audio = audio[:orig_len]
        elif len(audio) < orig_len:
            audio = np.pad(audio, (0, orig_len - len(audio)))

    # 2. Speaker + mic frequency response
    if bandpass:
        audio = apply_bandpass_filter(audio, sr, low_hz, high_hz)

    # 3. Ambient noise
    if snr_db < 100:
        audio = apply_additive_noise(audio, snr_db)

    # 4. Volume/distance
    if gain_db != 0.0:
        audio = apply_volume_scaling(audio, gain_db)

    return audio


# ============================================================================
# Empirical OTA Channel (from SoundCloud airgap recordings)
# ============================================================================

# Path to empirical channel artifacts
_EMPIRICAL_CHANNEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "soundcloud_recorded", "analysis"
)

_empirical_ir = None
_empirical_noise_bank = None


def _load_empirical_channel():
    """Lazy-load empirical IR and noise bank from disk."""
    global _empirical_ir, _empirical_noise_bank

    if _empirical_ir is not None:
        return

    ir_path = os.path.join(_EMPIRICAL_CHANNEL_DIR, "channel_impulse_response.npy")
    if not os.path.isfile(ir_path):
        raise FileNotFoundError(
            f"Empirical IR not found at {ir_path}. "
            "Run soundcloud_recorded/airgap_noise_analysis.ipynb first."
        )
    _empirical_ir = np.load(ir_path)

    # Load noise bank
    noise_files = sorted(
        f for f in os.listdir(_EMPIRICAL_CHANNEL_DIR)
        if f.startswith("residual_noise_") and f.endswith(".npy")
    )
    _empirical_noise_bank = [
        np.load(os.path.join(_EMPIRICAL_CHANNEL_DIR, f)) for f in noise_files
    ]


def simulate_empirical_ota_channel(
    audio_np: np.ndarray,
    sr: int,
    noise_scale: float = 0.0,
) -> np.ndarray:
    """
    Apply empirical OTA channel from MacBook Air → air → iPhone 12 Pro recordings.

    The channel is modeled as a linear convolution with the empirical impulse
    response H(f), extracted from paired original/recorded audio. This captures
    the content-independent speaker + air + mic frequency response.

    Optionally adds scaled residual noise, but the raw residuals contain
    content-dependent nonlinear distortion at ~-3.6 dB SNR, so noise_scale
    should be kept very low (0.0-0.05) or zero for realistic simulation.

    Args:
        audio_np: Input waveform (1D numpy array)
        sr: Sample rate (should be 24000 to match empirical data)
        noise_scale: Scale factor for residual noise (0.0 = IR only, recommended)

    Returns:
        Distorted audio, same length as input
    """
    from scipy import signal as sp_signal

    _load_empirical_channel()

    audio = audio_np.copy()

    # Convolve with empirical IR (linear channel: speaker + air + mic)
    audio = sp_signal.fftconvolve(audio, _empirical_ir, mode='full')[:len(audio_np)]

    # Gain-normalize to match original RMS
    orig_rms = np.sqrt(np.mean(audio_np ** 2) + 1e-10)
    conv_rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    audio = audio * (orig_rms / conv_rms)

    # Optionally add mild residual noise (scaled down from raw empirical level)
    if _empirical_noise_bank and noise_scale > 0:
        idx = np.random.randint(len(_empirical_noise_bank))
        noise = _empirical_noise_bank[idx]
        if len(noise) > len(audio):
            noise = noise[:len(audio)]
        elif len(noise) < len(audio):
            reps = (len(audio) // len(noise)) + 1
            noise = np.tile(noise, reps)[:len(audio)]
        audio = audio + noise_scale * noise

    return audio.astype(np.float32)


# ============================================================================
# Model Loading (reuses eval_models.py pattern)
# ============================================================================

def load_model(model_name: str, device: str = "cuda"):
    """Load an Audio LLM by name. Returns (model, sample_rate)."""
    if model_name == "qwen2_audio":
        from models.qwen2_audio import Qwen2AudioModel
        model = Qwen2AudioModel(
            model_path=MODEL_PATHS["qwen2_audio"],
            device=device, dtype=torch.bfloat16,
        )
        return model, TARGET_SAMPLE_RATE

    elif model_name == "qwen25_omni":
        from models.qwen25_omni import Qwen25OmniModel
        model = Qwen25OmniModel(
            model_path=MODEL_PATHS["qwen25_omni"],
            device=device, dtype=torch.bfloat16,
        )
        return model, TARGET_SAMPLE_RATE

    elif model_name == "audio_flamingo":
        from models.audio_flamingo import AudioFlamingoModel
        model = AudioFlamingoModel(
            model_path=MODEL_PATHS["audio_flamingo"],
            device=device, dtype=torch.bfloat16,
        )
        return model, model.SAMPLE_RATE

    elif model_name == "kimi_audio":
        from models.kimi_audio import KimiAudioModel
        model = KimiAudioModel(
            model_path=MODEL_PATHS.get("kimi_audio"),
            device=device,
        )
        return model, model.SAMPLE_RATE

    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_audio(model, model_name: str, audio_np: np.ndarray, sr: int,
                   prompt: str, device: str = "cuda") -> str:
    """Run inference on audio. Returns generated text."""
    wav_tensor = torch.FloatTensor(audio_np).unsqueeze(0).to(device)

    if model_name == "kimi_audio":
        # Kimi needs a file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_np, sr)
            output = model.generate_from_path(f.name, prompt=prompt)
            os.unlink(f.name)
    else:
        with torch.no_grad():
            output = model.generate(wav_tensor, prompt=prompt)

    return output


# ============================================================================
# Demo Modes
# ============================================================================

def demo_simulate_single(args, model, model_name, model_sr, prompt):
    """Simulate OTA on a single adversarial audio file."""
    print(f"\n{'='*70}")
    print(f"OTA SIMULATION — Single File")
    print(f"{'='*70}")
    print(f"Audio:  {args.audio}")
    print(f"Target: {args.target}")
    print(f"Model:  {model_name}")
    print(f"Channel: Opus {args.opus_bitrate}kbps | SNR {args.snr_db}dB | "
          f"Bandpass {args.low_hz}-{args.high_hz}Hz | Gain {args.gain_db}dB")
    print()

    # Load audio
    audio_np, sr = sf.read(args.audio)
    if sr != ENCODEC_SAMPLE_RATE:
        audio_tensor = torch.FloatTensor(audio_np).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, ENCODEC_SAMPLE_RATE)
        audio_np = audio_tensor.squeeze(0).numpy()
        sr = ENCODEC_SAMPLE_RATE

    # --- Baseline (no degradation) ---
    print("1) BASELINE (digital, no channel degradation)")
    if model_sr != sr:
        baseline_tensor = torchaudio.functional.resample(
            torch.FloatTensor(audio_np).unsqueeze(0), sr, model_sr
        )
        baseline_np = baseline_tensor.squeeze(0).numpy()
    else:
        baseline_np = audio_np
    baseline_output = evaluate_audio(model, model_name, baseline_np, model_sr, prompt)
    baseline_match = args.target.lower() in baseline_output.lower() if args.target else None
    print(f"   Output:  {baseline_output}")
    if args.target:
        print(f"   Match:   {'YES' if baseline_match else 'NO'}")
    print()

    # --- Opus only ---
    print(f"2) OPUS {args.opus_bitrate}kbps ONLY (simulates YouTube/Spotify upload)")
    opus_audio = apply_opus_compression(audio_np, sr, args.opus_bitrate)
    if len(opus_audio) > len(audio_np):
        opus_audio = opus_audio[:len(audio_np)]
    elif len(opus_audio) < len(audio_np):
        opus_audio = np.pad(opus_audio, (0, len(audio_np) - len(opus_audio)))
    if model_sr != sr:
        opus_tensor = torchaudio.functional.resample(
            torch.FloatTensor(opus_audio).unsqueeze(0), sr, model_sr
        )
        opus_np = opus_tensor.squeeze(0).numpy()
    else:
        opus_np = opus_audio
    opus_output = evaluate_audio(model, model_name, opus_np, model_sr, prompt)
    opus_match = args.target.lower() in opus_output.lower() if args.target else None
    print(f"   Output:  {opus_output}")
    if args.target:
        print(f"   Match:   {'YES' if opus_match else 'NO'}")
    print()

    # --- Full OTA simulation ---
    print(f"3) FULL OTA SIMULATION (Opus + bandpass + noise + volume)")
    ota_audio = simulate_ota_channel(
        audio_np, sr,
        opus_bitrate=args.opus_bitrate,
        snr_db=args.snr_db,
        bandpass=not args.no_bandpass,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        gain_db=args.gain_db,
    )
    if model_sr != sr:
        ota_tensor = torchaudio.functional.resample(
            torch.FloatTensor(ota_audio).unsqueeze(0), sr, model_sr
        )
        ota_np = ota_tensor.squeeze(0).numpy()
    else:
        ota_np = ota_audio
    ota_output = evaluate_audio(model, model_name, ota_np, model_sr, prompt)
    ota_match = args.target.lower() in ota_output.lower() if args.target else None
    print(f"   Output:  {ota_output}")
    if args.target:
        print(f"   Match:   {'YES' if ota_match else 'NO'}")
    print()

    # Save OTA-degraded audio for listening
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.audio))[0]
        ota_path = os.path.join(args.save_dir, f"{base}_ota.wav")
        sf.write(ota_path, ota_audio, sr)
        print(f"   Saved OTA audio: {ota_path}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':<40} {'Output (first 60 chars)':<60} {'Match':>5}")
    print("-" * 107)
    print(f"{'Digital (baseline)':<40} {baseline_output[:60]:<60} {'YES' if baseline_match else 'NO':>5}")
    print(f"{'Opus ' + str(args.opus_bitrate) + 'kbps':<40} {opus_output[:60]:<60} {'YES' if opus_match else 'NO':>5}")
    print(f"{'Full OTA':<40} {ota_output[:60]:<60} {'YES' if ota_match else 'NO':>5}")

    return {
        "baseline": {"output": baseline_output, "match": baseline_match},
        "opus": {"output": opus_output, "match": opus_match},
        "ota": {"output": ota_output, "match": ota_match},
    }


def demo_simulate_benchmark(args, model, model_name, model_sr, prompt):
    """Simulate OTA on all audio files in a benchmark directory."""
    from eval_models import find_audio_files, load_attack_records

    print(f"\n{'='*70}")
    print(f"OTA SIMULATION — Benchmark")
    print(f"{'='*70}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Model:     {model_name}")
    print(f"Channel:   Opus {args.opus_bitrate}kbps | SNR {args.snr_db}dB | "
          f"Bandpass {args.low_hz}-{args.high_hz}Hz")
    print()

    audio_files = find_audio_files(args.benchmark)
    attack_records = load_attack_records(args.benchmark)

    if not audio_files:
        print("ERROR: No audio files found.")
        return

    print(f"Found {len(audio_files)} adversarial audio files\n")

    results = []
    baseline_ok, opus_ok, ota_ok = 0, 0, 0

    for i, (exp_key, wav_path) in enumerate(sorted(audio_files.items())):
        rec = attack_records.get(exp_key, {})
        target_text = rec.get("target_text", rec.get("command_text", ""))

        print(f"[{i+1}/{len(audio_files)}] {exp_key}")

        # Load
        audio_np, sr = sf.read(wav_path)
        if sr != ENCODEC_SAMPLE_RATE:
            audio_t = torchaudio.functional.resample(
                torch.FloatTensor(audio_np).unsqueeze(0), sr, ENCODEC_SAMPLE_RATE
            )
            audio_np = audio_t.squeeze(0).numpy()
            sr = ENCODEC_SAMPLE_RATE

        def to_model_sr(np_audio):
            if model_sr != sr:
                t = torchaudio.functional.resample(
                    torch.FloatTensor(np_audio).unsqueeze(0), sr, model_sr
                )
                return t.squeeze(0).numpy()
            return np_audio

        # Baseline
        bl_out = evaluate_audio(model, model_name, to_model_sr(audio_np), model_sr, prompt)
        bl_match = target_text.lower() in bl_out.lower() if target_text else False

        # OTA
        ota_audio = simulate_ota_channel(
            audio_np, sr,
            opus_bitrate=args.opus_bitrate,
            snr_db=args.snr_db,
            bandpass=not args.no_bandpass,
            low_hz=args.low_hz,
            high_hz=args.high_hz,
            gain_db=args.gain_db,
        )
        ota_out = evaluate_audio(model, model_name, to_model_sr(ota_audio), model_sr, prompt)
        ota_match = target_text.lower() in ota_out.lower() if target_text else False

        if bl_match:
            baseline_ok += 1
        if ota_match:
            ota_ok += 1

        r = {
            "exp_key": exp_key,
            "target": target_text,
            "baseline": {"output": bl_out, "match": bl_match},
            "ota": {"output": ota_out, "match": ota_match},
        }
        results.append(r)

        bl_s = "OK" if bl_match else "FAIL"
        ota_s = "OK" if ota_match else "FAIL"
        print(f"  Baseline: {bl_s:<4} | OTA: {ota_s:<4} | {ota_out[:70]}")

    # Summary
    n = len(results)
    print(f"\n{'='*70}")
    print("OTA BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"Model:     {model_name}")
    print(f"Channel:   Opus {args.opus_bitrate}kbps + SNR {args.snr_db}dB + bandpass")
    print(f"Baseline:  {baseline_ok}/{n} ({100*baseline_ok/n:.1f}%)")
    print(f"After OTA: {ota_ok}/{n} ({100*ota_ok/n:.1f}%)")

    # Save results
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, f"ota_{model_name}_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(out_path, "w") as f:
            json.dump({
                "model": model_name,
                "channel": {
                    "opus_bitrate": args.opus_bitrate,
                    "snr_db": args.snr_db,
                    "bandpass": not args.no_bandpass,
                    "low_hz": args.low_hz,
                    "high_hz": args.high_hz,
                    "gain_db": args.gain_db,
                },
                "baseline_rate": baseline_ok / n,
                "ota_rate": ota_ok / n,
                "results": results,
            }, f, indent=2)
        print(f"Results saved: {out_path}")

    return results


def demo_record(args, model, model_name, model_sr, prompt):
    """Record audio from microphone and evaluate."""
    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: sounddevice not installed. Run: pip install sounddevice")
        return

    duration = args.duration
    record_sr = ENCODEC_SAMPLE_RATE

    print(f"\n{'='*70}")
    print(f"LIVE RECORDING — Microphone Input")
    print(f"{'='*70}")
    print(f"Model:    {model_name}")
    print(f"Duration: {duration}s")
    print(f"Sample rate: {record_sr} Hz")
    print()

    input("Press ENTER to start recording...")
    print(f"Recording for {duration}s...")

    audio_np = sd.rec(int(duration * record_sr), samplerate=record_sr,
                      channels=1, dtype="float32")
    sd.wait()
    audio_np = audio_np.squeeze()

    print("Recording complete.\n")

    # Save recording
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        rec_path = os.path.join(args.save_dir, f"recording_{datetime.now():%Y%m%d_%H%M%S}.wav")
        sf.write(rec_path, audio_np, record_sr)
        print(f"Saved recording: {rec_path}")

    # Resample to model SR
    if model_sr != record_sr:
        audio_tensor = torchaudio.functional.resample(
            torch.FloatTensor(audio_np).unsqueeze(0), record_sr, model_sr
        )
        audio_for_model = audio_tensor.squeeze(0).numpy()
    else:
        audio_for_model = audio_np

    output = evaluate_audio(model, model_name, audio_for_model, model_sr, prompt)
    print(f"\nModel output: {output}")
    if args.target:
        match = args.target.lower() in output.lower()
        print(f"Target: {args.target}")
        print(f"Match:  {'YES' if match else 'NO'}")


def demo_play_and_record(args, model, model_name, model_sr, prompt):
    """Play adversarial audio through speaker, record from mic, evaluate."""
    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: sounddevice not installed. Run: pip install sounddevice")
        return

    print(f"\n{'='*70}")
    print(f"PLAY & RECORD — Real Over-the-Air Demo")
    print(f"{'='*70}")
    print(f"Audio:  {args.audio}")
    print(f"Target: {args.target}")
    print(f"Model:  {model_name}")
    print()

    # Load adversarial audio
    audio_np, sr = sf.read(args.audio)
    duration = len(audio_np) / sr

    print(f"Audio duration: {duration:.1f}s at {sr} Hz")
    print()
    print("This will:")
    print("  1. Play the adversarial audio through your default speaker")
    print("  2. Simultaneously record from your default microphone")
    print("  3. Feed the recording to the Audio LLM")
    print()

    input("Press ENTER when ready (make sure speaker & mic are set up)...")

    # Record slightly longer than playback to capture full audio
    record_duration = duration + 1.0
    record_sr = ENCODEC_SAMPLE_RATE

    print(f"Playing & recording ({record_duration:.1f}s)...")

    # Start recording
    recording = sd.rec(int(record_duration * record_sr), samplerate=record_sr,
                       channels=1, dtype="float32")

    # Play audio (blocks until done)
    sd.play(audio_np, sr)
    sd.wait()

    # Wait for recording to finish
    time.sleep(1.0)
    sd.stop()
    recording = recording.squeeze()

    print("Done.\n")

    # Save
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.audio))[0]
        rec_path = os.path.join(args.save_dir, f"{base}_ota_recording.wav")
        sf.write(rec_path, recording, record_sr)
        print(f"Saved OTA recording: {rec_path}")

    # Evaluate
    if model_sr != record_sr:
        rec_tensor = torchaudio.functional.resample(
            torch.FloatTensor(recording).unsqueeze(0), record_sr, model_sr
        )
        rec_for_model = rec_tensor.squeeze(0).numpy()
    else:
        rec_for_model = recording

    output = evaluate_audio(model, model_name, rec_for_model, model_sr, prompt)
    print(f"\nModel output: {output}")
    if args.target:
        match = args.target.lower() in output.lower()
        print(f"Target: {args.target}")
        print(f"Match:  {'YES' if match else 'NO'}")


# ============================================================================
# SNR sweep — test multiple noise levels at once
# ============================================================================

def demo_snr_sweep(args, model, model_name, model_sr, prompt):
    """Sweep over multiple SNR levels to find the robustness threshold."""
    print(f"\n{'='*70}")
    print(f"SNR SWEEP — Robustness Threshold")
    print(f"{'='*70}")
    print(f"Audio:  {args.audio}")
    print(f"Target: {args.target}")
    print(f"Model:  {model_name}")
    print(f"Opus:   {args.opus_bitrate} kbps")
    print()

    audio_np, sr = sf.read(args.audio)
    if sr != ENCODEC_SAMPLE_RATE:
        audio_t = torchaudio.functional.resample(
            torch.FloatTensor(audio_np).unsqueeze(0), sr, ENCODEC_SAMPLE_RATE
        )
        audio_np = audio_t.squeeze(0).numpy()
        sr = ENCODEC_SAMPLE_RATE

    def to_model_sr(np_audio):
        if model_sr != sr:
            t = torchaudio.functional.resample(
                torch.FloatTensor(np_audio).unsqueeze(0), sr, model_sr
            )
            return t.squeeze(0).numpy()
        return np_audio

    snr_levels = [50, 40, 35, 30, 25, 20, 15, 10]

    print(f"{'SNR (dB)':<12} {'Output (first 70 chars)':<70} {'Match':>5}")
    print("-" * 89)

    # Baseline (no noise, no compression)
    bl_out = evaluate_audio(model, model_name, to_model_sr(audio_np), model_sr, prompt)
    bl_match = args.target.lower() in bl_out.lower() if args.target else False
    print(f"{'inf (clean)':<12} {bl_out[:70]:<70} {'YES' if bl_match else 'NO':>5}")

    for snr in snr_levels:
        ota_audio = simulate_ota_channel(
            audio_np, sr,
            opus_bitrate=args.opus_bitrate,
            snr_db=snr,
            bandpass=not args.no_bandpass,
            low_hz=args.low_hz,
            high_hz=args.high_hz,
            gain_db=0.0,
        )
        out = evaluate_audio(model, model_name, to_model_sr(ota_audio), model_sr, prompt)
        match = args.target.lower() in out.lower() if args.target else False
        print(f"{snr:<12} {out[:70]:<70} {'YES' if match else 'NO':>5}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Over-the-Air adversarial audio demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulate", action="store_true",
                      help="Simulate OTA channel on adversarial audio")
    mode.add_argument("--record", action="store_true",
                      help="Record from microphone and evaluate")
    mode.add_argument("--play-and-record", action="store_true",
                      help="Play audio through speaker, record from mic, evaluate")
    mode.add_argument("--snr-sweep", action="store_true",
                      help="Sweep over SNR levels to find robustness threshold")

    # Input
    parser.add_argument("--audio", type=str, help="Path to adversarial WAV file")
    parser.add_argument("--benchmark", type=str, help="Path to benchmark directory")
    parser.add_argument("--target", type=str, help="Target text to match")

    # Model
    parser.add_argument("--model", type=str, default="qwen2_audio",
                        choices=["qwen2_audio", "qwen25_omni", "audio_flamingo", "kimi_audio"],
                        help="Model to evaluate")
    parser.add_argument("--prompt-mode", type=str, default=None,
                        help="Prompt mode (transcribe/qa)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt text")

    # Channel parameters
    parser.add_argument("--opus-bitrate", type=int, default=128,
                        help="Opus compression bitrate in kbps (default: 128)")
    parser.add_argument("--snr-db", type=float, default=30.0,
                        help="Signal-to-noise ratio in dB (default: 30)")
    parser.add_argument("--no-bandpass", action="store_true",
                        help="Disable bandpass filter")
    parser.add_argument("--low-hz", type=float, default=100.0,
                        help="Bandpass lower cutoff in Hz (default: 100)")
    parser.add_argument("--high-hz", type=float, default=14000.0,
                        help="Bandpass upper cutoff in Hz (default: 14000)")
    parser.add_argument("--gain-db", type=float, default=-3.0,
                        help="Volume gain in dB (default: -3)")

    # Recording
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Recording duration in seconds (default: 10)")

    # Output
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save OTA audio and results")

    args = parser.parse_args()

    # Resolve prompt
    if args.prompt:
        prompt_text = args.prompt
    elif args.prompt_mode:
        prompt_text = PROMPT_MODES[args.prompt_mode]["prompt"]
    elif args.benchmark:
        config_path = os.path.join(args.benchmark, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config = json.load(f)
            prompt_text = config.get("prompt_text", PROMPT_MODES[DEFAULT_PROMPT_MODE]["prompt"])
        else:
            prompt_text = PROMPT_MODES[DEFAULT_PROMPT_MODE]["prompt"]
    else:
        prompt_text = PROMPT_MODES[DEFAULT_PROMPT_MODE]["prompt"]

    print(f"Loading model: {args.model}")
    print(f"Prompt: \"{prompt_text}\"")
    model, model_sr = load_model(args.model)

    if args.simulate:
        if args.benchmark:
            demo_simulate_benchmark(args, model, args.model, model_sr, prompt_text)
        elif args.audio:
            demo_simulate_single(args, model, args.model, model_sr, prompt_text)
        else:
            print("ERROR: --simulate requires --audio or --benchmark")
    elif args.record:
        demo_record(args, model, args.model, model_sr, prompt_text)
    elif args.play_and_record:
        if not args.audio:
            print("ERROR: --play-and-record requires --audio")
            return
        demo_play_and_record(args, model, args.model, model_sr, prompt_text)
    elif args.snr_sweep:
        if not args.audio:
            print("ERROR: --snr-sweep requires --audio")
            return
        demo_snr_sweep(args, model, args.model, model_sr, prompt_text)


if __name__ == "__main__":
    main()
