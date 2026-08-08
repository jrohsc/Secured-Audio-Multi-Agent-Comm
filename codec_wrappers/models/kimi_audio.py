"""
Kimi-Audio-7B-Instruct wrapper — INFERENCE ONLY (held-out victim).

Purpose: a fourth architecture to hold out from the ensemble-victim transfer
probe. With Qwen2-Audio + Qwen2.5-Omni + Audio-Flamingo-3 all serving as
ensemble MEMBERS, the only remaining held-out model was Omni-3B, which is the
same family as member Omni-7B — so the 3-member ensemble could not test
cross-architecture transfer at all. Kimi restores a genuine cross-architecture
held-out victim.

`compute_loss` is deliberately NOT implemented. A held-out victim is only ever
asked to generate text from a wav; it never needs a differentiable path. Making
Kimi an ensemble *member* would require soft-quantizing its discrete audio
tokenizer (cf. personaplex.py's soft-RVQ) — a different and much larger job.
Raising here keeps that distinction impossible to blur by accident.

Kimi's API takes an audio FILE PATH, not a tensor, so `generate` round-trips the
waveform through a temp 16 kHz wav. That is lossless for our purposes (the
delivery channel already rendered the audio) and is the only supported entry.
"""

import os
import sys
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from models.base import BaseAudioModel

# `kimia_infer` is not pip-installed; point KIMI_INFER_ROOT at a checkout of
# https://github.com/MoonshotAI/Kimi-Audio (see docs/REPRODUCE.md).
KIMI_INFER_ROOT = os.environ.get("KIMI_INFER_ROOT", "")

KIMI_SAMPLE_RATE = 16000


class KimiAudioModel(BaseAudioModel):
    """Inference-only Kimi-Audio victim. Requires the `kimi-audio` conda env."""

    def __init__(self, model_path: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16, load_detokenizer: bool = False):
        if KIMI_INFER_ROOT and KIMI_INFER_ROOT not in sys.path:
            sys.path.insert(0, KIMI_INFER_ROOT)
        try:
            from kimia_infer.api.kimia import KimiAudio
        except ImportError as e:
            raise ImportError(
                f"kimia_infer not importable from {KIMI_INFER_ROOT!r}. Run this in "
                f"the `kimi-audio` conda env, or set KIMI_INFER_ROOT."
            ) from e

        self._device = device
        # KimiAudio hardcodes bfloat16 internally and moves itself to
        # torch.cuda.current_device(); `dtype` is recorded for the interface but
        # does not change the loaded weights.
        self._dtype = dtype

        # load_detokenizer=False skips the audio-output vocoder (and its
        # first-run CUDA extension build). We only ever read text.
        self.model = KimiAudio(model_path=model_path, load_detokenizer=load_detokenizer)

        # Greedy, matching how the other wrappers score: deterministic text so a
        # rerun of the same clip cannot change the success verdict.
        self.sampling_params = {
            "text_temperature": 0.0,
            "text_top_k": 5,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }

    @property
    def sample_rate(self) -> int:
        return KIMI_SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def generate(self, wav: torch.Tensor, prompt: Optional[str] = None,
                 max_new_tokens: int = -1, **kwargs) -> str:
        """wav: float tensor [T] or [1, T] at 16 kHz. Returns the text answer."""
        audio = wav.detach().to(torch.float32).cpu()
        if audio.dim() == 2:
            if audio.shape[0] != 1:
                raise ValueError(f"expected mono [1, T], got {tuple(audio.shape)}")
            audio = audio.squeeze(0)
        elif audio.dim() != 1:
            raise ValueError(f"expected [T] or [1, T], got {tuple(audio.shape)}")

        messages = []
        if prompt:
            # Text instruction must precede the audio content — see the ordering
            # note in kimia_infer/api/kimia.py:generate.
            messages.append({"role": "user", "message_type": "text", "content": prompt})
        messages.append({"role": "user", "message_type": "audio", "content": None})

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            sf.write(tmp.name, audio.numpy().astype(np.float32), KIMI_SAMPLE_RATE)
            messages[-1]["content"] = tmp.name
            with torch.no_grad():
                _, text = self.model.generate(
                    messages,
                    output_type="text",
                    max_new_tokens=max_new_tokens,
                    **self.sampling_params,
                )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        return (text or "").strip()

    def compute_loss(self, wav: torch.Tensor, target_text: str, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "KimiAudioModel is inference-only: it exists to be a HELD-OUT victim. "
            "Using it as an ensemble member needs a differentiable path through "
            "Kimi's discrete audio tokenizer (see personaplex.py's soft-RVQ)."
        )
