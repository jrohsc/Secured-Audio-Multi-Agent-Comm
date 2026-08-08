"""
DAC (Descript Audio Codec) wrapper for latent-space adversarial attacks.

Mirrors EnCodecWrapper's interface (encode_to_continuous / decode_from_continuous)
so attack code is codec-agnostic. Operates on the pre-quantization continuous
latent so we can compute gradients through the decoder.

Reference: external/codecattack_lib/attacks/latent_codec.py (EnCodecWrapper)
API verified against descript-audio-codec 1.0.0.
"""

import torch
import dac


class DACWrapper:
    """
    Wrapper around Descript Audio Codec exposing the continuous pre-quantization
    latent space for gradient-based attacks. Same interface as EnCodecWrapper.
    """

    def __init__(self, model_type: str = "24khz", device: str = "cuda"):
        self.device = device
        self.sample_rate = 24000  # DAC 24kHz model
        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path)
        self.model.to(device)
        self.model.train(False)  # inference mode

    def encode_to_continuous(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to continuous pre-quantization latent.

        Args:
            wav: [B, 1, T] at 24kHz

        Returns:
            z: [B, D, T_z], pre-quantization continuous latent.
               D depends on the DAC model (typically 1024 for 24khz).
        """
        with torch.no_grad():
            z = self.model.encoder(wav)
        return z

    def decode_from_continuous(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode audio from continuous latent. Differentiable when
        self.model.decoder is in train() mode.

        Args:
            z: [B, D, T_z]

        Returns:
            wav: [B, 1, T] at 24kHz
        """
        audio = self.model.decoder(z)
        return audio
