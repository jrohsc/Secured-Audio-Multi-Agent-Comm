"""
AAC/MP3 codec augmentation with Straight-Through Estimator (STE) for
channel-robust adversarial attacks.

Uses ffmpeg AAC encode/decode as a non-differentiable forward pass,
with gradients passing through as identity (BPDA). This simulates
the real SoundCloud transcoding pipeline more faithfully than Opus
or EnCodec proxies.

Supports multiple codec profiles for EoT:
  - AAC at various bitrates (64, 96, 128, 192, 256 kbps)
  - MP3 at various bitrates (for generalization)
"""

import os
import sys
import shutil
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_ffmpeg() -> str:
    """Find ffmpeg binary."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        env_bin = os.path.dirname(sys.executable)
        ffmpeg_bin = os.path.join(env_bin, "ffmpeg")
        if not os.path.isfile(ffmpeg_bin):
            raise FileNotFoundError("ffmpeg not found")
    return ffmpeg_bin


def _make_tempdir_resilient() -> str:
    # Cluster nodes occasionally have a flaky /tmp where mkdtemp races against
    # cleanup and raises FileNotFoundError. Retry a few times, then fall back
    # to a stable per-user scratch dir under $HOME.
    for attempt in range(4):
        try:
            return tempfile.mkdtemp()
        except FileNotFoundError:
            time.sleep(0.2 * (2 ** attempt))
    home_scratch = os.path.join(os.path.expanduser("~"), ".cache", "codec_attack_scratch")
    os.makedirs(home_scratch, exist_ok=True)
    return tempfile.mkdtemp(dir=home_scratch)


def apply_aac_compression(
    audio_np: np.ndarray,
    sr: int,
    bitrate_kbps: int = 128,
    ffmpeg_bin: str = None,
) -> np.ndarray:
    """Compress audio through AAC codec via ffmpeg (simulates SoundCloud)."""
    if ffmpeg_bin is None:
        ffmpeg_bin = _find_ffmpeg()

    tmpdir = _make_tempdir_resilient()
    try:
        in_path = os.path.join(tmpdir, "input.wav")
        aac_path = os.path.join(tmpdir, "compressed.m4a")
        out_path = os.path.join(tmpdir, "output.wav")

        sf.write(in_path, audio_np, sr)

        # Encode to AAC
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", in_path,
             "-c:a", "aac", "-b:a", f"{bitrate_kbps}k",
             "-ar", str(sr), aac_path],
            capture_output=True, check=True,
        )
        # Decode back to WAV
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", aac_path,
             "-c:a", "pcm_f32le", "-ar", str(sr), out_path],
            capture_output=True, check=True,
        )

        decoded, _ = sf.read(out_path, dtype="float32")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return decoded


def apply_soundcloud_compression(
    audio_np: np.ndarray,
    sr: int,
    bitrate_kbps: int = 160,
    ffmpeg_bin: str = None,
    sc_sr: int = 44100,
    add_encoder_padding: bool = True,
    padding_samples_44k: int = 2048,
) -> np.ndarray:
    """Simulate full SoundCloud pipeline: upsample -> AAC -> downsample.

    Real SoundCloud chain:
      24kHz mono -> 44.1kHz stereo -> AAC 160k -> 44.1kHz stereo -> mono -> 24kHz

    Real SoundCloud also adds ~2048 samples of encoder padding at 44.1kHz
    (~46ms) at the start, which shifts the mel spectrogram and breaks
    adversarial alignment. This is simulated when add_encoder_padding=True.
    """
    if ffmpeg_bin is None:
        ffmpeg_bin = _find_ffmpeg()

    tmpdir = _make_tempdir_resilient()
    try:
        in_path = os.path.join(tmpdir, "input.wav")
        aac_path = os.path.join(tmpdir, "compressed.m4a")
        out_path = os.path.join(tmpdir, "output.wav")

        sf.write(in_path, audio_np, sr)

        # Encode: upsample to sc_sr, stereo, AAC at target bitrate
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", in_path,
             "-ac", "2", "-ar", str(sc_sr),
             "-c:a", "aac", "-b:a", f"{bitrate_kbps}k",
             aac_path],
            capture_output=True, check=True,
        )
        # Decode: back to mono at original sr
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", aac_path,
             "-ac", "1", "-ar", str(sr),
             "-c:a", "pcm_f32le", out_path],
            capture_output=True, check=True,
        )

        decoded, _ = sf.read(out_path, dtype="float32")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Simulate encoder padding: prepend silence equivalent to
    # padding_samples_44k samples at 44.1kHz, converted to output sr
    if add_encoder_padding and padding_samples_44k > 0:
        pad_at_sr = int(padding_samples_44k * sr / sc_sr)
        decoded = np.concatenate([np.zeros(pad_at_sr, dtype=decoded.dtype), decoded])

    return decoded


def apply_mp3_compression(
    audio_np: np.ndarray,
    sr: int,
    bitrate_kbps: int = 128,
    ffmpeg_bin: str = None,
) -> np.ndarray:
    """Compress audio through MP3 codec via ffmpeg."""
    if ffmpeg_bin is None:
        ffmpeg_bin = _find_ffmpeg()

    tmpdir = _make_tempdir_resilient()
    try:
        in_path = os.path.join(tmpdir, "input.wav")
        mp3_path = os.path.join(tmpdir, "compressed.mp3")
        out_path = os.path.join(tmpdir, "output.wav")

        sf.write(in_path, audio_np, sr)

        subprocess.run(
            [ffmpeg_bin, "-y", "-i", in_path,
             "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k",
             "-ar", str(sr), mp3_path],
            capture_output=True, check=True,
        )
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", mp3_path,
             "-c:a", "pcm_f32le", "-ar", str(sr), out_path],
            capture_output=True, check=True,
        )

        decoded, _ = sf.read(out_path, dtype="float32")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return decoded


class AACCodecSTE(nn.Module):
    """
    AAC/MP3 codec augmentation with Straight-Through Estimator.

    Forward pass: audio_tensor -> numpy -> ffmpeg AAC/MP3 -> numpy -> tensor
    Backward pass: gradients pass through as identity (STE / BPDA)

    For EoT, randomly samples codec type and bitrate each forward pass.
    """

    # SoundCloud typically uses AAC at 128-256 kbps
    DEFAULT_AAC_BITRATES = [96, 128, 192, 256]
    DEFAULT_MP3_BITRATES = [128, 192]

    def __init__(
        self,
        sample_rate: int = 24000,
        aac_bitrates: List[int] = None,
        mp3_bitrates: List[int] = None,
        include_mp3: bool = False,
        mp3_only: bool = False,
        soundcloud_mode: bool = False,
        soundcloud_sr: int = 44100,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.aac_bitrates = aac_bitrates or self.DEFAULT_AAC_BITRATES
        self.mp3_bitrates = mp3_bitrates or self.DEFAULT_MP3_BITRATES
        self.include_mp3 = include_mp3
        self.mp3_only = mp3_only
        self.soundcloud_mode = soundcloud_mode
        self.soundcloud_sr = soundcloud_sr
        self._ffmpeg_bin = _find_ffmpeg()

        # Build codec pool for EoT
        self._codec_pool: List[Tuple[str, int]] = []
        if mp3_only:
            # MP3-only EoT (codec-EoT ablation): pool is exclusively MP3.
            for br in self.mp3_bitrates:
                self._codec_pool.append(("mp3", br))
        elif soundcloud_mode:
            # SoundCloud serves AAC 160k — also include nearby bitrates for EoT
            for br in self.aac_bitrates:
                self._codec_pool.append(("soundcloud", br))
        else:
            for br in self.aac_bitrates:
                self._codec_pool.append(("aac", br))
        if include_mp3 and not mp3_only:
            for br in self.mp3_bitrates:
                self._codec_pool.append(("mp3", br))

    def _apply_codec(self, audio_np: np.ndarray, codec: str, bitrate: int) -> np.ndarray:
        if codec == "soundcloud":
            return apply_soundcloud_compression(
                audio_np, self.sample_rate, bitrate, self._ffmpeg_bin,
                sc_sr=self.soundcloud_sr,
            )
        elif codec == "aac":
            return apply_aac_compression(
                audio_np, self.sample_rate, bitrate, self._ffmpeg_bin
            )
        else:
            return apply_mp3_compression(
                audio_np, self.sample_rate, bitrate, self._ffmpeg_bin
            )

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        """
        Apply AAC/MP3 codec with STE.

        Args:
            x: Audio tensor [B, 1, T] at sample_rate
            severity: 0=bypass, 1=full codec

        Returns:
            Codec-degraded audio with STE gradient path.
        """
        if severity < 0.01:
            return x

        # Randomly pick codec and bitrate (EoT)
        idx = torch.randint(len(self._codec_pool), (1,)).item()
        codec_type, bitrate = self._codec_pool[idx]

        # Non-differentiable codec forward pass
        with torch.no_grad():
            audio_np = x.squeeze(0).squeeze(0).cpu().numpy()
            degraded_np = self._apply_codec(audio_np, codec_type, bitrate)

            # Match length
            if len(degraded_np) > len(audio_np):
                degraded_np = degraded_np[:len(audio_np)]
            elif len(degraded_np) < len(audio_np):
                degraded_np = np.pad(degraded_np, (0, len(audio_np) - len(degraded_np)))

            degraded = torch.from_numpy(degraded_np).float().to(x.device)
            degraded = degraded.view_as(x)

        # STE: forward uses degraded, backward treats as identity
        result = x + (degraded - x).detach()

        if severity < 1.0:
            result = severity * result + (1.0 - severity) * x

        return result
