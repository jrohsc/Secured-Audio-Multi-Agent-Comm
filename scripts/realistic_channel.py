"""
Realistic SoundCloud+airgap channel simulation for robust adversarial attacks.

Based on empirical measurements from 7 original/recorded pairs (eps=5.0, calm_1).
Captures the actual frequency response of the full chain:
  SoundCloud (AAC + loudness norm) -> MacBook speaker -> air -> iPhone mic

Key findings from measurement:
  - Channel passband: ~176-8889 Hz (-6dB points)
  - Peak boost at ~5kHz (+16 dB) — speaker/mic resonance
  - Heavy low-freq rolloff: -17dB at 100Hz
  - High-freq rolloff: -22dB at 11kHz
  - Overall RMS loss: ~42% (0.58 ratio)
  - Nonlinear: RMS varies per sample (AGC/compression)

Components (all differentiable or STE):
  1. Loudness normalization (LUFS-style) — SoundCloud preprocessing
  2. Empirical frequency shaping — measured transfer function via FIR filter
  3. Soft clipping — speaker nonlinearity
  4. Additive noise — room ambient (from background_empty_noise.m4a)
  5. Gain variation — mic AGC randomization
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as scipy_signal


class LoudnessNormalizer(nn.Module):
    """
    LUFS-style loudness normalization (SoundCloud preprocessing).
    Normalizes RMS to a target level. Differentiable.
    """
    def __init__(self, target_db: float = -14.0, randomize_db: float = 2.0):
        super().__init__()
        self.target_db = target_db
        self.randomize_db = randomize_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target = self.target_db
        if self.training and self.randomize_db > 0:
            target += (torch.rand(1).item() - 0.5) * 2 * self.randomize_db

        rms = torch.sqrt(torch.mean(x ** 2) + 1e-10)
        current_db = 20 * torch.log10(rms + 1e-10)
        gain_db = target - current_db
        gain = 10 ** (gain_db / 20)
        return x * gain


class EmpiricalFrequencyShaping(nn.Module):
    """
    Apply the measured channel transfer function as an FIR filter.

    Loads the empirical transfer function (power spectrum ratio) from
    real SoundCloud+airgap recordings and designs an FIR filter that
    applies the same frequency shaping. Fully differentiable via F.conv1d.
    """
    def __init__(
        self,
        sample_rate: int = 24000,
        num_taps: int = 257,
        tf_path: str = None,
        tf_freqs_path: str = None,
        randomize: float = 0.15,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_taps = num_taps
        self.randomize = randomize

        if tf_path and os.path.isfile(tf_path):
            tf_power = np.load(tf_path)
            tf_freqs = np.load(tf_freqs_path) if tf_freqs_path else None
            self._design_from_tf(tf_power, tf_freqs)
        else:
            # Fallback: approximate MacBook+iPhone channel
            self._design_approximate()

    def _design_from_tf(self, tf_power, tf_freqs):
        """Design FIR filter from measured transfer function."""
        # tf_power is PSD ratio (linear), convert to amplitude
        tf_amplitude = np.sqrt(np.maximum(tf_power, 1e-10))

        # Interpolate to uniform frequency grid for firwin2
        nyq = self.sample_rate / 2.0
        if tf_freqs is not None:
            # Normalize frequencies to [0, 1] for firwin2
            norm_freqs = tf_freqs / nyq
        else:
            norm_freqs = np.linspace(0, 1, len(tf_amplitude))

        # Clip to valid range
        valid = (norm_freqs >= 0) & (norm_freqs <= 1.0)
        norm_freqs = norm_freqs[valid]
        tf_amplitude = tf_amplitude[valid]

        # Ensure endpoints
        if norm_freqs[0] > 0:
            norm_freqs = np.concatenate([[0], norm_freqs])
            tf_amplitude = np.concatenate([[tf_amplitude[0]], tf_amplitude])
        if norm_freqs[-1] < 1.0:
            norm_freqs = np.concatenate([norm_freqs, [1.0]])
            tf_amplitude = np.concatenate([tf_amplitude, [tf_amplitude[-1]]])

        # Clamp extreme values (don't amplify more than 20dB or attenuate more than 30dB)
        tf_amplitude = np.clip(tf_amplitude, 10**(-30/20), 10**(20/20))

        # Design FIR filter
        taps = scipy_signal.firwin2(self.num_taps, norm_freqs, tf_amplitude)
        self.register_buffer("default_taps", torch.FloatTensor(taps))

        # Pre-compute randomized filter bank
        bank = [torch.FloatTensor(taps)]
        for _ in range(15):
            jitter = 1.0 + (np.random.rand(len(tf_amplitude)) - 0.5) * 2 * self.randomize
            jittered = tf_amplitude * jitter
            jittered = np.clip(jittered, 10**(-30/20), 10**(20/20))
            t = scipy_signal.firwin2(self.num_taps, norm_freqs, jittered)
            bank.append(torch.FloatTensor(t))
        self.register_buffer("filter_bank", torch.stack(bank))

    def _design_approximate(self):
        """Fallback: approximate MacBook speaker + iPhone mic response."""
        nyq = self.sample_rate / 2.0
        # Approximate measured response
        freqs = np.array([0, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 11000, 12000]) / nyq
        # dB gains from measurement
        gains_db = np.array([-20, -17, -9, 4, -14, 6, 3, 1, 11, 16, -1, 5, -21, -23, -30])
        gains = 10 ** (gains_db / 20)
        gains = np.clip(gains, 10**(-30/20), 10**(20/20))

        taps = scipy_signal.firwin2(self.num_taps, freqs, gains)
        self.register_buffer("default_taps", torch.FloatTensor(taps))
        self.register_buffer("filter_bank", torch.FloatTensor(taps).unsqueeze(0))

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        if severity < 0.01:
            return x

        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        if self.training and self.filter_bank.shape[0] > 1:
            idx = torch.randint(self.filter_bank.shape[0], (1,)).item()
            taps = self.filter_bank[idx]
        else:
            taps = self.default_taps

        kernel = taps.view(1, 1, -1).to(x.device, x.dtype)
        pad = self.num_taps // 2
        filtered = F.conv1d(x, kernel, padding=pad)
        filtered = filtered[..., :x.shape[-1]]

        # Match RMS to avoid pure gain changes
        orig_rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-10)
        filt_rms = torch.sqrt(torch.mean(filtered ** 2, dim=-1, keepdim=True) + 1e-10)
        filtered = filtered * (orig_rms / filt_rms)

        if severity < 1.0:
            filtered = severity * filtered + (1.0 - severity) * x

        if squeeze:
            filtered = filtered.squeeze(0)
        return filtered


class SpeakerNonlinearity(nn.Module):
    """
    Simulate speaker distortion with soft clipping + harmonics.
    MacBook speakers are small and clip significantly at high volumes.
    """
    def __init__(self, drive: float = 2.0, mix: float = 0.3):
        super().__init__()
        self.drive = drive
        self.mix = mix

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        if severity < 0.01:
            return x

        drive = self.drive * severity
        if self.training:
            drive *= (0.8 + 0.4 * torch.rand(1).item())

        # Soft clip via tanh
        clipped = torch.tanh(x * drive) / drive

        # Blend
        mix = self.mix * severity
        result = (1 - mix) * x + mix * clipped
        return result


class MicAGC(nn.Module):
    """
    Simulate iPhone mic automatic gain control.
    Applies random gain variation (the mic adapts to signal level).
    """
    def __init__(self, gain_std_db: float = 3.0):
        super().__init__()
        self.gain_std_db = gain_std_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            gain_db = torch.randn(1).item() * self.gain_std_db
            gain = 10 ** (gain_db / 20)
            return x * gain
        return x


class RealisticOTAChannel(nn.Module):
    """
    Full realistic SoundCloud+airgap channel.

    Pipeline:
      1. Loudness normalization (SoundCloud LUFS targeting)
      2. Empirical frequency shaping (speaker + air + mic response)
      3. Speaker soft clipping (nonlinearity)
      4. Mic AGC (random gain)
      5. Additive noise (room ambient, detached)

    All components use differentiable torch ops (or STE for noise).
    Gradients flow through the entire chain except noise.
    """
    def __init__(
        self,
        sample_rate: int = 24000,
        tf_path: str = None,
        tf_freqs_path: str = None,
        bg_noise_path: str = None,
        noise_snr_db: float = 25.0,
    ):
        super().__init__()

        self.loudness = LoudnessNormalizer(target_db=-14.0, randomize_db=2.0)

        self.freq_shape = EmpiricalFrequencyShaping(
            sample_rate=sample_rate,
            tf_path=tf_path,
            tf_freqs_path=tf_freqs_path,
        )

        self.speaker_clip = SpeakerNonlinearity(drive=2.0, mix=0.3)
        self.agc = MicAGC(gain_std_db=3.0)
        self.noise_snr_db = noise_snr_db

        # Load background noise
        self._bg_noise = None
        if bg_noise_path and os.path.isfile(bg_noise_path):
            self._load_noise(bg_noise_path, sample_rate)

    def _load_noise(self, path, sr):
        import subprocess, tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp = f.name
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(sr),
                 '-acodec', 'pcm_f32le', tmp],
                capture_output=True, check=True,
            )
            data, _ = sf.read(tmp)
            self._bg_noise = torch.FloatTensor(data)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self._bg_noise is None:
            return x

        T = x.shape[-1]
        noise = self._bg_noise.to(x.device)
        if len(noise) < T:
            noise = noise.repeat((T // len(noise)) + 1)
        offset = torch.randint(0, max(1, len(noise) - T), (1,)).item()
        noise_seg = noise[offset:offset + T]

        snr = self.noise_snr_db
        if self.training:
            snr += torch.randn(1).item() * 5.0  # randomize

        sig_power = torch.mean(x.detach() ** 2)
        noise_power = torch.mean(noise_seg ** 2) + 1e-10
        target_power = sig_power / (10 ** (snr / 10) + 1e-10)
        scale = torch.sqrt(target_power / noise_power)

        return x + (noise_seg * scale).detach().view_as(x)

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        """Apply full realistic OTA channel."""
        # 1. Loudness normalization
        y = self.loudness(x)
        # 2. Frequency shaping (speaker + air + mic)
        y = self.freq_shape(y, severity=severity)
        # 3. Speaker nonlinearity
        y = self.speaker_clip(y, severity=severity)
        # 4. Mic AGC
        y = self.agc(y)
        # 5. Ambient noise
        y = self._add_noise(y)
        return y


class PhysicalChannelFilter(nn.Module):
    """
    Differentiable FIR filter matching the measured physical OTA transfer function.

    Designed from PSD ratios of adversarial audio vs iPhone recordings — captures
    the actual frequency shaping (bass rolloff, 2-8kHz boost) rather than a
    Wiener-deconvolved IR which is signal-dependent for nonlinear channels.

    Robust mode adds per-band random gain jitter to cover the ~40% stochastic
    variance that a static FIR can't capture (AGC, multipath, room variation).
    """
    def __init__(
        self,
        fir_path: str = None,
        sample_rate: int = 24000,
        jitter: float = 0.1,
        # Robust mode: per-band random gain jitter
        robust: bool = False,
        band_jitter_db: float = 3.0,
        n_bands: int = 8,
        gain_jitter_db: float = 3.0,
        # Phase randomization (simulates phase destruction from reflections)
        phase_jitter: float = 0.0,  # max phase jitter in radians (0=disabled, pi=full random)
    ):
        super().__init__()
        if fir_path is None:
            fir_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data", "empirical_ota", "physical_channel_fir.npy",
            )
        if not os.path.isfile(fir_path):
            raise FileNotFoundError(f"Physical channel FIR not found: {fir_path}")

        taps = np.load(fir_path).astype(np.float32)
        self.register_buffer("fir_taps", torch.FloatTensor(taps))
        self.num_taps = len(taps)
        self.jitter = jitter
        self.robust = robust
        self.band_jitter_db = band_jitter_db
        self.n_bands = n_bands
        self.gain_jitter_db = gain_jitter_db
        self.sample_rate = sample_rate
        self.phase_jitter = phase_jitter
        print(f"PhysicalChannelFilter: loaded {self.num_taps}-tap FIR from {fir_path}"
              f"{' [robust mode]' if robust else ''}"
              f"{f' [phase_jitter={phase_jitter:.2f} rad]' if phase_jitter > 0 else ''}")

    def _apply_phase_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random per-band phase rotation in frequency domain (differentiable).

        The physical channel inverts phase in some bands (correlation ~ -0.3).
        This simulates that by adding smooth random phase offsets per band.
        """
        T = x.shape[-1]
        X = torch.fft.rfft(x)
        n_freq = X.shape[-1]

        # Generate smooth random phase offsets
        # Use fewer control points than frequency bins for smooth variation
        n_control = self.n_bands
        control_phases = (torch.rand(n_control, device=x.device) - 0.5) * 2 * self.phase_jitter

        # Interpolate to full frequency resolution
        phase_offsets = torch.nn.functional.interpolate(
            control_phases.view(1, 1, -1),
            size=n_freq,
            mode='linear',
            align_corners=True,
        ).view(-1)

        # Apply phase rotation: X * e^(j*phase)
        cos_p = torch.cos(phase_offsets).to(x.dtype)
        sin_p = torch.sin(phase_offsets).to(x.dtype)
        X_real = X.real * cos_p - X.imag * sin_p
        X_imag = X.real * sin_p + X.imag * cos_p
        X = torch.complex(X_real, X_imag)

        return torch.fft.irfft(X, n=T)

    def _apply_band_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random per-band gain in frequency domain (differentiable)."""
        T = x.shape[-1]
        # FFT
        X = torch.fft.rfft(x)
        n_freq = X.shape[-1]

        # Create per-band random gains
        band_edges = torch.linspace(0, n_freq, self.n_bands + 1, device=x.device).long()
        gains_db = (torch.rand(self.n_bands, device=x.device) - 0.5) * 2 * self.band_jitter_db
        gains_linear = 10.0 ** (gains_db / 20.0)

        # Build smooth gain curve via interpolation
        gain_curve = torch.ones(n_freq, device=x.device, dtype=x.dtype)
        for i in range(self.n_bands):
            lo, hi = band_edges[i].item(), band_edges[i + 1].item()
            gain_curve[lo:hi] = gains_linear[i].to(x.dtype)

        # Smooth with a small moving average to avoid sharp transitions
        kernel_size = max(3, n_freq // 32)
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        gain_curve_smooth = F.avg_pool1d(
            gain_curve.view(1, 1, -1),
            kernel_size=kernel_size, stride=1, padding=pad,
        ).view(-1)[:n_freq]

        X = X * gain_curve_smooth
        return torch.fft.irfft(X, n=T)

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        if severity < 0.01:
            return x

        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze = True

        taps = self.fir_taps
        if self.training and self.jitter > 0:
            # Global gain variation per call (simulates channel non-stationarity)
            gain_jitter = 1.0 + (torch.rand(1, device=x.device) - 0.5) * 2 * self.jitter
            taps = taps * gain_jitter

        kernel = taps.view(1, 1, -1).to(x.device, x.dtype)
        pad = self.num_taps // 2
        y = F.conv1d(x, kernel, padding=pad)[..., :x.shape[-1]]

        # Robust mode: per-band random gain jitter + phase jitter + global gain
        if self.training and self.robust:
            y = self._apply_band_jitter(y)
            # Phase randomization (simulates phase destruction from reflections)
            if self.phase_jitter > 0:
                y = self._apply_phase_jitter(y)
            # Additional global gain variation (simulates distance/volume)
            gain_db = (torch.rand(1, device=x.device) - 0.5) * 2 * self.gain_jitter_db
            y = y * (10.0 ** (gain_db / 20.0))

        if severity < 1.0:
            y = severity * y + (1.0 - severity) * x

        if squeeze:
            y = y.squeeze(0)
            if y.dim() > 1:
                y = y.squeeze(0)

        return y


class EmpiricalNonlinearity(nn.Module):
    """
    Data-driven per-band polynomial nonlinearity fitted from real
    MacBook Air -> iPhone 16 Pro OTA recordings.

    For each frequency band, applies: y_band = a1*x + a2*x^2 + a3*x^3
    where coefficients are extracted from paired recordings via
    extract_nonlinearity.py.

    Differentiable via torch ops (bandpass via conv1d, polynomial via
    element-wise ops). Gradients flow through all bands.
    """
    def __init__(
        self,
        model_path: str = None,
        sample_rate: int = 24000,
        num_taps: int = 129,
        jitter: float = 0.1,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_taps = num_taps
        self.jitter = jitter  # randomize coefficients during training

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data", "empirical_ota", "nonlinearity_model.json",
            )

        import json
        with open(model_path) as f:
            model = json.load(f)

        # Build per-band filters and coefficients
        self._bands = []
        nyq = sample_rate / 2.0

        for band in model['bands']:
            if band.get('skipped', False):
                continue
            # Only use bands with significant nonlinearity (dR2 > 0.01)
            if band['improvement'] < 0.01:
                continue

            low, high = band['low'], band['high']
            coeffs = band['coeffs']  # [dc, a1, a2, a3, ...]

            # Design bandpass FIR filter
            if low <= 0:
                b = scipy_signal.firwin(num_taps, high / nyq, pass_zero=True)
            elif high >= nyq:
                b = scipy_signal.firwin(num_taps, low / nyq, pass_zero=False)
            else:
                b = scipy_signal.firwin(
                    num_taps, [low / nyq, high / nyq], pass_zero=False,
                )

            self._bands.append({
                'low': low, 'high': high,
                'name': f"{low}-{high}Hz",
            })
            idx = len(self._bands) - 1
            self.register_buffer(f"bp_filter_{idx}",
                                 torch.FloatTensor(b).view(1, 1, -1))
            self.register_buffer(f"coeffs_{idx}",
                                 torch.FloatTensor(coeffs))

        print(f"EmpiricalNonlinearity: loaded {len(self._bands)} active bands "
              f"from {model_path}")
        for i, bd in enumerate(self._bands):
            c = getattr(self, f"coeffs_{i}")
            print(f"  {bd['name']}: coeffs={[f'{v:.4f}' for v in c.tolist()]}")

    def _apply_bandpass(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Apply bandpass filter for band idx. Input: [B, 1, T]."""
        filt = getattr(self, f"bp_filter_{idx}")
        filt = filt.to(x.device, x.dtype)
        pad = self.num_taps // 2
        return F.conv1d(x, filt, padding=pad)[..., :x.shape[-1]]

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        if severity < 0.01 or len(self._bands) == 0:
            return x

        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze = True

        # Start with the input, replace each band with its nonlinear version
        y = torch.zeros_like(x)
        covered = torch.zeros_like(x)

        for idx in range(len(self._bands)):
            x_band = self._apply_bandpass(x, idx)
            coeffs = getattr(self, f"coeffs_{idx}").to(x.device, x.dtype)

            # Optionally jitter coefficients during training for robustness
            if self.training and self.jitter > 0:
                jitter = 1.0 + (torch.rand(len(coeffs), device=x.device) - 0.5) * 2 * self.jitter
                coeffs = coeffs * jitter

            # Apply polynomial: y = dc + a1*x + a2*x^2 + a3*x^3 + ...
            y_band = coeffs[0]  # DC offset
            for k in range(1, len(coeffs)):
                # Odd-symmetric for odd powers, even-symmetric for even
                if k % 2 == 0:
                    y_band = y_band + coeffs[k] * x_band.abs().pow(k) * x_band.sign()
                else:
                    y_band = y_band + coeffs[k] * x_band.pow(k)

            y = y + y_band
            covered = covered + x_band

        # Add back uncovered frequencies (pass-through)
        uncovered = x - covered
        y = y + uncovered

        # Blend with original based on severity
        if severity < 1.0:
            y = severity * y + (1.0 - severity) * x

        if squeeze:
            y = y.squeeze(0)
            if y.dim() > 1:
                y = y.squeeze(0)

        return y


class FrequencyShapedPerturbation(nn.Module):
    """
    Shapes the adversarial perturbation to concentrate energy in
    frequency bands that survive the physical channel.

    Based on measurement: the channel passes 500Hz-8kHz well.
    Energy below 200Hz or above 10kHz is wasted.

    Applied as a differentiable bandpass filter on the perturbation
    (not the signal), so the optimizer learns to put energy where
    the channel preserves it.
    """
    def __init__(self, sample_rate: int = 24000, low_hz: float = 400.0,
                 high_hz: float = 8000.0, num_taps: int = 129):
        super().__init__()
        nyq = sample_rate / 2.0
        low = max(low_hz / nyq, 0.001)
        high = min(high_hz / nyq, 0.999)
        taps = scipy_signal.firwin(num_taps, [low, high], pass_zero=False)
        self.register_buffer("kernel", torch.FloatTensor(taps).view(1, 1, -1))
        self.pad = num_taps // 2

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Filter perturbation to channel-survivable band."""
        if delta.dim() == 2:
            delta = delta.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        filtered = F.conv1d(delta, self.kernel.to(delta.device, delta.dtype),
                           padding=self.pad)
        filtered = filtered[..., :delta.shape[-1]]

        if squeeze:
            filtered = filtered.squeeze(0)
        return filtered
