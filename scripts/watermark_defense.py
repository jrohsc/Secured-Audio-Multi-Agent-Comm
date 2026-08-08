"""
AudioSeal watermark defense against latent-space adversarial attacks.

Defense concept:
  Sender: watermark(audio) -> encode(codec) -> transmit
  Receiver: decode(codec) -> verify_watermark(audio) -> accept/reject -> LLM

Since adversarial perturbations target the LLM's behavior (not watermark
preservation), they should corrupt the watermark, enabling detection.

AudioSeal operates at 16kHz; EnCodec at 24kHz. This module handles resampling.
"""

import torch
import torchaudio
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class WatermarkResult:
    """Result of watermark detection on an audio sample."""

    detection_score: float  # 0-1, higher = more likely watermarked
    message_accuracy: float  # 0-1, fraction of message bits correctly decoded
    is_watermarked: bool  # detection_score >= threshold
    threshold: float  # detection threshold used

    def to_json(self) -> dict:
        return {
            "detection_score": self.detection_score,
            "message_accuracy": self.message_accuracy,
            "is_watermarked": self.is_watermarked,
            "threshold": self.threshold,
        }


class AudioSealDefense:
    """
    Watermark-based defense using Meta's AudioSeal.

    Embeds imperceptible watermarks into audio before codec encoding.
    After decoding, verifies watermark presence to authenticate audio.
    Adversarial perturbations (not optimized for watermark preservation)
    are expected to corrupt the watermark, enabling detection and rejection.
    """

    AUDIOSEAL_SR = 16000  # AudioSeal native sample rate

    def __init__(
        self,
        nbits: int = 16,
        detection_threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.nbits = nbits
        self.detection_threshold = detection_threshold
        self.device = device

        # Load AudioSeal models
        from audioseal import AudioSeal

        self.generator = AudioSeal.load_generator(f"audioseal_wm_{nbits}bits").to(device)
        self.detector = AudioSeal.load_detector(f"audioseal_detector_{nbits}bits").to(device)

        self.generator.eval()
        self.detector.eval()

    def embed_watermark(
        self,
        audio_16k: torch.Tensor,
        message: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed watermark into 16kHz audio.

        Args:
            audio_16k: [B, 1, T] at 16kHz
            message: Optional [B, nbits] binary tensor. Random if None.

        Returns:
            (watermarked_16k [B, 1, T], message [B, nbits])
        """
        audio_16k = audio_16k.to(self.device)
        if audio_16k.dim() == 2:
            audio_16k = audio_16k.unsqueeze(1)  # [B, T] -> [B, 1, T]

        if message is None:
            message = torch.randint(0, 2, (audio_16k.shape[0], self.nbits), device=self.device)

        with torch.no_grad():
            watermarked = self.generator(audio_16k, sample_rate=self.AUDIOSEAL_SR, message=message)

        return watermarked, message

    def detect_watermark(
        self,
        audio_16k: torch.Tensor,
        expected_message: Optional[torch.Tensor] = None,
    ) -> WatermarkResult:
        """
        Detect watermark in 16kHz audio.

        Args:
            audio_16k: [B, 1, T] at 16kHz
            expected_message: Optional [B, nbits] to check message accuracy

        Returns:
            WatermarkResult with detection score and message accuracy
        """
        audio_16k = audio_16k.to(self.device)
        if audio_16k.dim() == 2:
            audio_16k = audio_16k.unsqueeze(1)

        with torch.no_grad():
            result = self.detector.detect_watermark(audio_16k, sample_rate=self.AUDIOSEAL_SR, message_threshold=self.detection_threshold)

        # result is (detection_score, decoded_message) per AudioSeal API
        detection_score = result[0].mean().item()  # average over batch/time

        # Message accuracy
        msg_accuracy = 0.0
        if expected_message is not None and result[1] is not None:
            decoded_msg = result[1]  # [B, nbits]
            msg_accuracy = (decoded_msg == expected_message).float().mean().item()

        is_watermarked = detection_score >= self.detection_threshold

        return WatermarkResult(
            detection_score=detection_score,
            message_accuracy=msg_accuracy,
            is_watermarked=is_watermarked,
            threshold=self.detection_threshold,
        )

    def detect_frame_level(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """
        Per-frame watermark detection scores.

        Args:
            audio_16k: [B, 1, T] at 16kHz

        Returns:
            [B, T_frames] detection scores
        """
        audio_16k = audio_16k.to(self.device)
        if audio_16k.dim() == 2:
            audio_16k = audio_16k.unsqueeze(1)

        with torch.no_grad():
            result = self.detector(audio_16k, sample_rate=self.AUDIOSEAL_SR)

        # result shape: [B, 2, T_frames] — channel 0 = detection logits
        return result[:, 0, :]  # [B, T_frames]

    def watermark_for_codec(
        self,
        audio_24k: torch.Tensor,
        message: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sender pipeline: embed watermark in 24kHz audio for codec transmission.

        Resamples 24k -> 16k, embeds watermark, resamples back 16k -> 24k.

        Args:
            audio_24k: [B, 1, T] at 24kHz (or [1, 1, T] for EnCodec)
            message: Optional message bits

        Returns:
            (watermarked_24k same shape as input, message [B, nbits])
        """
        # Handle EnCodec [1, 1, T] shape
        squeeze_batch = False
        if audio_24k.dim() == 3 and audio_24k.shape[0] == 1 and audio_24k.shape[1] == 1:
            # [1, 1, T] -> treat as B=1, C=1, T
            pass
        elif audio_24k.dim() == 2:
            audio_24k = audio_24k.unsqueeze(1)

        batch_size = audio_24k.shape[0]

        # Resample 24k -> 16k
        audio_16k = torchaudio.functional.resample(
            audio_24k.squeeze(1), 24000, self.AUDIOSEAL_SR
        )  # [B, T_16k]
        audio_16k = audio_16k.unsqueeze(1)  # [B, 1, T_16k]

        # Embed watermark
        watermarked_16k, message = self.embed_watermark(audio_16k, message)

        # Resample back 16k -> 24k
        watermarked_24k = torchaudio.functional.resample(
            watermarked_16k.squeeze(1), self.AUDIOSEAL_SR, 24000
        )  # [B, T_24k]
        watermarked_24k = watermarked_24k.unsqueeze(1)  # [B, 1, T_24k]

        # Match original length
        orig_len = audio_24k.shape[-1]
        if watermarked_24k.shape[-1] > orig_len:
            watermarked_24k = watermarked_24k[..., :orig_len]
        elif watermarked_24k.shape[-1] < orig_len:
            pad = orig_len - watermarked_24k.shape[-1]
            watermarked_24k = torch.nn.functional.pad(watermarked_24k, (0, pad))

        return watermarked_24k, message

    def verify_after_codec(
        self,
        audio_24k: torch.Tensor,
        expected_message: Optional[torch.Tensor] = None,
    ) -> WatermarkResult:
        """
        Receiver pipeline: verify watermark in decoded 24kHz audio.

        Resamples 24k -> 16k, then runs detection.

        Args:
            audio_24k: [B, 1, T] or [B, T] at 24kHz
            expected_message: Expected message bits for accuracy check

        Returns:
            WatermarkResult
        """
        if audio_24k.dim() == 2:
            audio_24k = audio_24k.unsqueeze(1)

        # Resample 24k -> 16k
        audio_16k = torchaudio.functional.resample(
            audio_24k.squeeze(1), 24000, self.AUDIOSEAL_SR
        )
        audio_16k = audio_16k.unsqueeze(1)  # [B, 1, T_16k]

        return self.detect_watermark(audio_16k, expected_message)
