"""
Music carrier loading and perceptual constraints for latent-space attacks.

Handles:
- Loading music from MP3/WAV files
- Resampling to EnCodec's 24kHz native rate
- Differentiable mel-spectrogram distance for perceptual loss
- Adaptive epsilon based on music loudness envelope
"""

import os
import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple

from config import (
    MUSIC_DIR, MUSIC_FILES, ENCODEC_SAMPLE_RATE, MUSIC_DURATION,
    MUSIC_DURATION_MIN, MUSIC_SECS_PER_WORD, PROJECT_ROOT,
)


def duration_for_target(target_text: str, max_duration: float = 10.0) -> float:
    """Compute music duration based on target word count.

    Longer targets need longer music to have enough audio capacity.
    Returns at least MUSIC_DURATION_MIN seconds, capped at max_duration
    (default 10s to avoid GPU OOM with EoT x4 during robust attack;
    increase to 15s+ if using EoT=1 or gradient accumulation).
    """
    word_count = len(target_text.split())
    needed = word_count * MUSIC_SECS_PER_WORD
    return min(max_duration, max(MUSIC_DURATION_MIN, needed))


def load_music(
    path: str,
    target_sr: int = ENCODEC_SAMPLE_RATE,
    duration: float = MUSIC_DURATION,
    normalize: bool = True
) -> torch.Tensor:
    """
    Load a music file, resample, and trim/pad to desired duration.

    Supports MP3, WAV, FLAC, etc. via librosa.

    Args:
        path: Path to audio file
        target_sr: Target sample rate (default: 24000 for EnCodec)
        duration: Duration in seconds to trim/pad to
        normalize: Whether to normalize audio to [-0.95, 0.95]

    Returns:
        Audio tensor [1, 1, T] ready for EnCodec encoder
    """
    import librosa

    wav_np, _ = librosa.load(path, sr=target_sr, mono=True, duration=duration)

    # Pad if shorter than desired duration
    target_length = int(duration * target_sr)
    if len(wav_np) < target_length:
        wav_np = np.pad(wav_np, (0, target_length - len(wav_np)), mode='constant')
    elif len(wav_np) > target_length:
        wav_np = wav_np[:target_length]

    wav = torch.FloatTensor(wav_np)

    if normalize:
        max_val = torch.max(torch.abs(wav))
        if max_val > 0:
            wav = wav / (max_val + 1e-8) * 0.95

    # Shape: [1, 1, T] for EnCodec
    return wav.unsqueeze(0).unsqueeze(0)


def resolve_music_path(name: str) -> Optional[str]:
    """
    Resolve a music argument to an absolute file path.

    Tries: absolute path, relative to CWD, relative to the repo root.
    Returns the resolved path or None if not found.
    """
    if name in MUSIC_FILES:
        return MUSIC_FILES[name]
    # Check if it looks like a path
    if os.path.sep in name or name.endswith(('.mp3', '.wav', '.flac', '.ogg')):
        # Try as-is (absolute or relative to CWD)
        if os.path.isfile(name):
            return os.path.abspath(name)
        # Try relative to the repo root
        candidate = os.path.join(PROJECT_ROOT, name)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def load_music_by_name(name: str, **kwargs) -> torch.Tensor:
    """
    Load a music file by its short name or file path.

    Args:
        name: Short name (e.g., "jazz_1") or path to an audio file (.mp3, .wav, etc.)
        **kwargs: Passed to load_music()

    Returns:
        Audio tensor [1, 1, T]
    """
    path = resolve_music_path(name)
    if path is None:
        available = ", ".join(MUSIC_FILES.keys())
        raise ValueError(f"Unknown music '{name}'. Not a known name ({available}) and not a valid file path.")
    return load_music(path, **kwargs)


def mel_distance(
    audio_a: torch.Tensor,
    audio_b: torch.Tensor,
    sample_rate: int = ENCODEC_SAMPLE_RATE,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256
) -> torch.Tensor:
    """
    Compute L2 distance in mel-spectrogram domain (differentiable).

    Used as a perceptual constraint to keep adversarial music sounding
    similar to the original.

    Args:
        audio_a: First audio tensor [1, T] or [1, 1, T]
        audio_b: Second audio tensor [1, T] or [1, 1, T]
        sample_rate: Audio sample rate
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        Scalar loss tensor (L2 distance in mel domain)
    """
    # Flatten to [1, T]
    if audio_a.dim() == 3:
        audio_a = audio_a.squeeze(0)
    if audio_b.dim() == 3:
        audio_b = audio_b.squeeze(0)

    # Ensure same length
    min_len = min(audio_a.shape[-1], audio_b.shape[-1])
    audio_a = audio_a[..., :min_len]
    audio_b = audio_b[..., :min_len]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=1.0,
    ).to(audio_a.device)

    mel_a = torch.log1p(mel_transform(audio_a))
    mel_b = torch.log1p(mel_transform(audio_b))

    return torch.nn.functional.mse_loss(mel_a, mel_b)


def multi_scale_mel_distance(
    audio_a: torch.Tensor,
    audio_b: torch.Tensor,
    sample_rate: int = ENCODEC_SAMPLE_RATE,
    fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
    n_mels: int = 80
) -> torch.Tensor:
    """
    Multi-scale spectral loss for better perceptual quality.

    Averages mel distance across multiple FFT sizes to capture
    both fine-grained and coarse spectral features.

    Args:
        audio_a: First audio tensor
        audio_b: Second audio tensor
        sample_rate: Audio sample rate
        fft_sizes: Tuple of FFT sizes to compute
        n_mels: Number of mel bands

    Returns:
        Averaged multi-scale mel distance
    """
    total_loss = torch.tensor(0.0, device=audio_a.device)
    for n_fft in fft_sizes:
        hop_length = n_fft // 4
        total_loss = total_loss + mel_distance(
            audio_a, audio_b,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
    return total_loss / len(fft_sizes)


def compute_adaptive_eps(
    music_wav: torch.Tensor,
    base_eps: float = 10.0,
    frame_size: int = 1024
) -> torch.Tensor:
    """
    Compute per-frame adaptive epsilon based on music loudness envelope.

    Louder frames can tolerate larger perturbations (masking effect).
    Quieter frames need smaller perturbations to remain imperceptible.

    Args:
        music_wav: Music waveform [1, 1, T]
        base_eps: Base epsilon value
        frame_size: Frame size for energy computation

    Returns:
        Per-frame epsilon tensor
    """
    if music_wav.dim() == 3:
        wav = music_wav.squeeze(0).squeeze(0)
    elif music_wav.dim() == 2:
        wav = music_wav.squeeze(0)
    else:
        wav = music_wav

    # Compute frame-level energy
    n_frames = wav.shape[0] // frame_size
    frames = wav[:n_frames * frame_size].reshape(n_frames, frame_size)
    energy = torch.sqrt(torch.mean(frames ** 2, dim=1))

    # Normalize to [0.3, 1.0] range (quiet frames still get some budget)
    energy_norm = energy / (energy.max() + 1e-8)
    energy_scaled = 0.3 + 0.7 * energy_norm

    # Scale by base_eps
    adaptive_eps = base_eps * energy_scaled

    return adaptive_eps
