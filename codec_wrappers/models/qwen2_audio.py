"""
Qwen2-Audio model wrapper for adversarial attacks.

Uses a differentiable MEL-level approach:
1. WAV -> MEL spectrogram (differentiable via torchaudio)
2. MEL -> audio features via model's audio encoder
3. Audio features + text embeddings -> inputs_embeds
4. Forward pass with inputs_embeds for loss computation
"""

import os
import torch
import torchaudio
from typing import Optional

from models.base import BaseAudioModel


class Qwen2AudioModel(BaseAudioModel):
    """
    Wrapper for Qwen2-Audio model with differentiable loss computation.

    Unlike Qwen2.5-Omni, Qwen2-Audio uses a different architecture
    (Qwen2AudioForConditionalGeneration) and requires special handling
    for the audio tower and multi-modal projector.
    """

    SAMPLE_RATE = 16000
    MEL_LENGTH = 3000  # Qwen2-Audio expects exactly 3000 mel frames

    # Local snapshot via $MODEL_PATH_QWEN2_AUDIO, else the public HF repo.
    DEFAULT_MODEL_PATH = os.environ.get(
        "MODEL_PATH_QWEN2_AUDIO", "Qwen/Qwen2-Audio-7B-Instruct"
    )

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        token: Optional[str] = None
    ):
        self._device = device
        self._dtype = dtype

        if model_path is None:
            model_path = self.DEFAULT_MODEL_PATH

        # Check if it's a local path
        local_files_only = os.path.isdir(model_path)

        print(f"Loading Qwen2-Audio model from: {model_path}")
        print(f"Device: {device}, Dtype: {dtype}")

        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
            token=token
        )

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            local_files_only=local_files_only,
            trust_remote_code=True,
            token=token,
            low_cpu_mem_usage=True
        ).eval()

        # Get the tokenizer
        self.tokenizer = self.processor.tokenizer

        # Store mel filters from processor's WhisperFeatureExtractor
        # for differentiable mel computation matching the processor exactly
        import numpy as np
        mel_filters_np = np.array(self.processor.feature_extractor.mel_filters)
        # mel_filters_np shape: (n_freqs=201, n_mels=128) -> transpose to (128, 201)
        self.mel_filters = torch.from_numpy(mel_filters_np.T).float().to(device)
        self.hann_window = torch.hann_window(400).to(device)

        print("Qwen2-Audio model loaded successfully!")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # Target waveform length: 30s * 16kHz = 480000 samples
    WAV_LENGTH = 480000

    def wav_to_mel(self, wav: torch.Tensor, pad_to_length: int = None) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram (differentiable).

        Matches WhisperFeatureExtractor exactly by zero-padding the raw
        waveform to 30s BEFORE computing STFT (not padding mel after).

        Args:
            wav: Audio waveform [1, T] or [T]
            pad_to_length: Pad mel to this length (default: MEL_LENGTH)

        Returns:
            Log-mel spectrogram [1, n_mels, T']
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self._device, dtype=torch.float32)

        # Zero-pad raw waveform to 30s (matches processor behavior exactly)
        wav_1d = wav.squeeze(0)
        if wav_1d.shape[0] < self.WAV_LENGTH:
            pad_zeros = torch.zeros(
                self.WAV_LENGTH - wav_1d.shape[0],
                dtype=wav_1d.dtype, device=wav_1d.device
            )
            wav_1d = torch.cat([wav_1d, pad_zeros])
        elif wav_1d.shape[0] > self.WAV_LENGTH:
            wav_1d = wav_1d[:self.WAV_LENGTH]

        # STFT -> magnitude squared (power=2.0)
        stft = torch.stft(
            wav_1d, n_fft=400, hop_length=160,
            window=self.hann_window, return_complex=True
        )
        magnitudes = stft.abs() ** 2  # [n_freqs, T]

        # Apply mel filterbank (from processor's WhisperFeatureExtractor)
        mel_spec = self.mel_filters @ magnitudes  # [128, T]

        # Whisper-style log normalization (differentiable)
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # Drop last frame (matches processor behavior)
        log_spec = log_spec[:, :-1]

        # Add batch dimension: [1, 128, T]
        mel = log_spec.unsqueeze(0)

        # Truncate if needed (should be exactly MEL_LENGTH=3000 now)
        if pad_to_length is None:
            pad_to_length = self.MEL_LENGTH
        if mel.shape[2] > pad_to_length:
            mel = mel[:, :, :pad_to_length]

        return mel

    DEFAULT_PROMPT = "What does the person say in this audio?"

    def generate(
        self,
        wav: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False,
        prompt: str = None,
        max_new_tokens: int = None,
    ) -> str:
        """
        Generate text from audio.

        Uses the processor's standard pipeline (non-differentiable).

        Args:
            wav: Audio waveform tensor [1, T]
            max_tokens: Maximum tokens to generate
            prompt: Text prompt to use (default: transcription prompt)
            max_new_tokens: Alias for max_tokens (for caller compatibility)

        Returns:
            Generated text string
        """
        if prompt is None:
            prompt = self.DEFAULT_PROMPT
        if max_new_tokens is not None:
            max_tokens = max_new_tokens

        with torch.no_grad():
            wav_np = wav.detach().squeeze().cpu().numpy()

            # Create conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": "placeholder"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Process inputs
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )

            audios = [wav_np]
            inputs = self.processor(
                text=text,
                audio=audios,
                return_tensors="pt",
                padding=True,
                sampling_rate=self.SAMPLE_RATE
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
            )

            # Decode only new tokens
            gen_ids = gen_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        return response.strip()

    # Cache for tokenized inputs — avoids repeated processor/tokenizer
    # calls and GPU↔CPU transfers in the optimization loop.
    _loss_cache_key = None
    _loss_cache = None

    def _get_cached_inputs(self, wav: torch.Tensor, target_text: str, prompt: str):
        """Return cached (full_input_ids, full_attention_mask, feature_attention_mask, labels).

        The processor output depends only on prompt text and audio length
        (for feature_attention_mask), not on audio content.  We cache by
        (prompt, target_text, wav_length) so repeated calls in an EoT /
        PGD loop skip the expensive processor + CPU round-trip entirely.
        """
        wav_len = wav.shape[-1]
        cache_key = (prompt, target_text, wav_len)

        if self._loss_cache_key == cache_key and self._loss_cache is not None:
            return self._loss_cache

        # --- cold path: run processor once ---
        wav_np = wav.detach().squeeze().cpu().numpy()
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "placeholder"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text, audio=[wav_np],
            return_tensors="pt", sampling_rate=self.SAMPLE_RATE
        )

        input_ids = inputs['input_ids'].to(self._device)
        attention_mask = inputs['attention_mask'].to(self._device)
        feature_attention_mask = inputs['feature_attention_mask'].to(self._device)

        target_ids = self.tokenizer(
            target_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._device)

        full_input_ids = torch.cat([input_ids, target_ids], dim=1)
        full_attention_mask = torch.cat([
            attention_mask,
            torch.ones_like(target_ids)
        ], dim=1)

        labels = torch.full_like(full_input_ids, -100)
        labels[0, -target_ids.shape[1]:] = target_ids[0]

        self._loss_cache_key = cache_key
        self._loss_cache = (full_input_ids, full_attention_mask,
                            feature_attention_mask, labels)
        return self._loss_cache

    def reset_cache(self):
        """Clear the tokenization cache (call when prompt/target changes)."""
        self._loss_cache_key = None
        self._loss_cache = None

    def compute_loss(
        self,
        wav: torch.Tensor,
        target_text: str,
        prompt: str = None,
        spec_augment: bool = False,
        spec_augment_n_mask: int = 10,
        spec_augment_mask_size: int = 50,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for target text (differentiable).

        Uses the model's own forward() method with:
        - input_ids from processor (with expanded audio tokens)
        - Differentiable mel features as input_features
        - feature_attention_mask from processor
        - Labels on target text tokens only

        This ensures the loss path matches the generation path exactly.

        Args:
            wav: Audio waveform tensor [1, T]
            target_text: Target text
            prompt: Text prompt to use (default: transcription prompt)
            spec_augment: If True, apply SpecAugment (random frequency band
                masking) on mel features before feeding to the model.
                Forces perturbation to distribute information across
                multiple frequency bands for OTA robustness.
            spec_augment_n_mask: Number of frequency bands to mask
            spec_augment_mask_size: Max width of each masked band (in mel bins)

        Returns:
            Loss tensor (scalar, with grad_fn)
        """
        if prompt is None:
            prompt = self.DEFAULT_PROMPT

        # 1. Compute differentiable mel features
        mel = self.wav_to_mel(wav, pad_to_length=self.MEL_LENGTH)

        # 1b. SpecAugment: randomly zero out frequency bands in mel
        # This is fully differentiable — gradients flow through unmasked bands.
        # Forces the adversarial perturbation to spread information across
        # many frequency bands, so even if some are destroyed by the physical
        # OTA channel, enough survive for the attack to succeed.
        if spec_augment:
            mel = mel.clone()
            n_mels = mel.shape[1]  # 128
            for _ in range(spec_augment_n_mask):
                f = torch.randint(0, max(1, spec_augment_mask_size), (1,)).item()
                f0 = torch.randint(0, max(1, n_mels - f), (1,)).item()
                mel[:, f0:f0 + f, :] = 0.0

        input_features = mel.to(self._dtype)  # [1, 128, 3000]

        # 2. Get tokenized inputs (cached across calls with same prompt/target/length)
        full_input_ids, full_attention_mask, feature_attention_mask, labels = \
            self._get_cached_inputs(wav, target_text, prompt)

        # 3. Forward pass through the FULL model (handles audio token
        #    replacement, attention masking, projector, etc. identically
        #    to generation)
        outputs = self.model(
            input_ids=full_input_ids,
            input_features=input_features,
            attention_mask=full_attention_mask,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
        )

        return outputs.loss
