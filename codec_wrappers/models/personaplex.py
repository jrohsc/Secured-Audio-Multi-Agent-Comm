"""
PersonaPLEX (Moshi-based) model wrapper for adversarial attacks.

Uses a soft-RVQ approach to bypass the non-differentiable discrete quantization:
1. Mimi encoder → continuous VQ latents z [B, 256, T]
2. Soft-quantize: compute soft distances to codebook vectors → soft weights
3. Use soft weights to look up Moshi embed_tokens → differentiable embeddings
4. Forward through Moshi decoder with text labels → CE loss

This enables gradient-based optimization in Mimi's continuous latent space
while maintaining a differentiable path to Moshi's text output loss.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from models.base import BaseAudioModel


class PersonaPLEXModel(BaseAudioModel):
    """
    Wrapper for Moshi/PersonaPLEX with differentiable soft-RVQ loss computation.

    Architecture:
    - Audio codec: Mimi (24kHz, 12.5fps, 256-dim VQ space)
    - Language model: Moshi (4096-dim, 32k text vocab, 8 codebooks)
    - Audio embedding: sum of 8 codebook embeddings (user) + 8 (moshi) + text

    The soft-RVQ trick makes the discrete codebook lookup differentiable by
    replacing argmin with softmax over L2 distances.
    """

    SAMPLE_RATE = 24000
    NUM_CODEBOOKS = 8  # Moshi uses 8 of Mimi's 32 codebooks

    DEFAULT_MODEL_PATH = "kmhf/hf-moshiko"

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        token: Optional[str] = None,
    ):
        self._device = device
        self._dtype = dtype

        if model_path is None:
            model_path = self.DEFAULT_MODEL_PATH

        local_files_only = os.path.isdir(model_path)

        print(f"Loading PersonaPLEX/Moshi model from: {model_path}")
        print(f"Device: {device}, Dtype: {dtype}")

        from transformers import MoshiForConditionalGeneration

        self.model = MoshiForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
            local_files_only=local_files_only,
            trust_remote_code=True,
            token=token,
            low_cpu_mem_usage=True,
        ).eval()

        # Get the text tokenizer (SentencePiece-based)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
            token=token,
        )

        # Cache key model dimensions
        self.hidden_size = self.model.config.hidden_size  # 4096
        self.audio_vocab_size = self.model.config.audio_vocab_size  # 2048
        self.text_vocab_size = self.model.config.vocab_size  # 32000

        # Extract Mimi codec reference (for encoding user audio in generate/compute_loss)
        self.mimi = self.model.audio_encoder

        # Extract and cache codebook vectors [codebook_size, codebook_dim]
        # from the Mimi quantizer for soft-RVQ
        self._codebook_vectors = self._extract_codebook_vectors()

        print(f"PersonaPLEX/Moshi loaded. Hidden: {self.hidden_size}, "
              f"Codebooks: {self.NUM_CODEBOOKS}, VQ dim: {self._codebook_vectors[0].shape}")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _extract_codebook_vectors(self) -> List[torch.Tensor]:
        """
        Extract codebook embedding vectors from Mimi's split RVQ.

        Codebook 0 comes from the semantic RVQ, codebooks 1-7 from acoustic RVQ.
        All vectors are in the 256-dim VQ space (post input_proj).

        Returns:
            List of K tensors, each [codebook_size, codebook_dim] = [2048, 256]
        """
        codebooks = []
        quantizer = self.mimi.quantizer

        # Codebook 0: semantic RVQ (1 layer)
        semantic_rvq = quantizer.semantic_residual_vector_quantizer
        for layer in semantic_rvq.layers:
            codebooks.append(layer.codebook.embed)  # [2048, 256]
            if len(codebooks) >= self.NUM_CODEBOOKS:
                return codebooks

        # Codebooks 1-7: acoustic RVQ
        acoustic_rvq = quantizer.acoustic_residual_vector_quantizer
        for layer in acoustic_rvq.layers:
            codebooks.append(layer.codebook.embed)  # [2048, 256]
            if len(codebooks) >= self.NUM_CODEBOOKS:
                return codebooks

        return codebooks

    def compute_loss_from_latents(
        self,
        z: torch.Tensor,
        target_text: str,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute differentiable CE loss from Mimi encoder latents using soft-RVQ.

        Gradient path:
            z → input_proj(512→256) → soft_distances → softmax
            → weighted_sum(Moshi_embeddings) → Moshi decoder → text logits → CE loss

        Args:
            z: Continuous encoder latent [B, 512, T] (pre-quantization, post-downsample)
            target_text: Target text the model should output
            temperature: Softmax temperature for soft codebook assignment
                (lower = closer to hard assignment, higher = smoother gradients)

        Returns:
            Loss tensor (scalar, with grad_fn)
        """
        B, C, T = z.shape

        # Apply input_proj to map from encoder space (512) to VQ space (256)
        # The semantic and acoustic RVQs have separate input_proj weights.
        # We apply the semantic input_proj for codebook 0, and acoustic for 1-7.
        quantizer = self.mimi.quantizer
        sem_rvq = quantizer.semantic_residual_vector_quantizer
        aco_rvq = quantizer.acoustic_residual_vector_quantizer

        # Project to VQ space using semantic input_proj (both use same dimensions)
        if sem_rvq.input_proj is not None:
            z_vq_sem = sem_rvq.input_proj(z)  # [B, 256, T]
        else:
            z_vq_sem = z
        if aco_rvq.input_proj is not None:
            z_vq_aco = aco_rvq.input_proj(z)  # [B, 256, T]
        else:
            z_vq_aco = z

        # Work in [B, T, 256] for distance computation
        z_sem_t = z_vq_sem.permute(0, 2, 1).float()  # [B, T, 256]
        z_aco_t = z_vq_aco.permute(0, 2, 1).float()  # [B, T, 256]

        # =====================================================================
        # Soft-RVQ: compute soft codebook assignments for each of 8 codebooks
        # =====================================================================
        user_audio_embeds = torch.zeros(
            B, T, self.hidden_size, device=z.device, dtype=z.dtype
        )

        # Codebook 0: semantic (uses z_sem_t)
        residual_sem = z_sem_t.clone()
        # Codebooks 1-7: acoustic (uses z_aco_t)
        residual_aco = z_aco_t.clone()

        for k in range(self.NUM_CODEBOOKS):
            cb_vectors = self._codebook_vectors[k].float()  # [2048, 256]

            if k == 0:
                residual = residual_sem
            else:
                residual = residual_aco

            # L2 distances: [B, T, 2048]
            dists = torch.cdist(residual, cb_vectors.unsqueeze(0).expand(B, -1, -1))

            # Soft assignment via softmax over negative distances
            soft_weights = F.softmax(-dists / temperature, dim=-1)  # [B, T, 2048]

            # Update residual with soft quantized value
            soft_quantized = torch.matmul(soft_weights, cb_vectors)  # [B, T, 256]
            if k == 0:
                residual_sem = residual_sem - soft_quantized
            else:
                residual_aco = residual_aco - soft_quantized

            # Look up Moshi user audio embeddings using soft weights
            # embed_tokens layout: [0..7] = moshi audio, [8..15] = user audio
            user_embed_idx = self.NUM_CODEBOOKS + k  # indices 8-15
            embed_weight = self.model.embed_tokens[user_embed_idx].weight  # [2049, 4096]
            # Only use first 2048 entries (last is padding/BOS token)
            embed_weight_audio = embed_weight[:self.audio_vocab_size].to(soft_weights.dtype)  # [2048, 4096]

            # Soft embedding lookup: [B, T, 2048] @ [2048, 4096] → [B, T, 4096]
            soft_embed = torch.matmul(soft_weights.to(embed_weight_audio.dtype), embed_weight_audio)
            user_audio_embeds = user_audio_embeds + soft_embed.to(user_audio_embeds.dtype)

        # =====================================================================
        # Build unconditional moshi audio + text embeddings
        # =====================================================================

        # Moshi audio: use unconditional (padding) tokens for all 8 moshi codebooks
        moshi_pad_code = self.audio_vocab_size  # padding token index
        moshi_codes = torch.full(
            (B, self.NUM_CODEBOOKS, T), moshi_pad_code,
            device=z.device, dtype=torch.long,
        )
        moshi_audio_embeds = sum(
            self.model.embed_tokens[k](moshi_codes[:, k])
            for k in range(self.NUM_CODEBOOKS)
        )  # [B, T, 4096]

        # Text: use padding token (vocab_size) for the full sequence
        text_pad_id = self.model.config.vocab_size  # special padding token
        text_ids = torch.full(
            (B, T), text_pad_id,
            device=z.device, dtype=torch.long,
        )
        text_embeds = self.model.decoder.model.embed_tokens(text_ids)  # [B, T, 4096]

        # Combined input embeddings (matches Moshi forward: audio + text)
        inputs_embeds = user_audio_embeds + moshi_audio_embeds + text_embeds  # [B, T, 4096]

        # =====================================================================
        # Tokenize target text and create labels
        # =====================================================================
        target_ids = self.tokenizer(
            target_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(z.device)  # [1, L]

        # Append target text embeddings to the sequence
        target_embeds = self.model.decoder.model.embed_tokens(target_ids)  # [1, L, 4096]
        # Also need "silence" audio embeddings during the target text portion
        # Both moshi audio (codebooks 0-7) and user audio (codebooks 8-15) get padding tokens
        L = target_ids.shape[1]
        pad_codes_single = torch.full(
            (B, L), moshi_pad_code, device=z.device, dtype=torch.long,
        )
        target_audio_embeds = sum(
            self.model.embed_tokens[k](pad_codes_single)
            for k in range(2 * self.NUM_CODEBOOKS)  # all 16 embed_tokens
        )  # [B, L, 4096]

        target_combined = target_embeds + target_audio_embeds  # [B, L, 4096]
        full_embeds = torch.cat([inputs_embeds, target_combined], dim=1)  # [B, T+L, 4096]

        # Labels: -100 for audio portion, target_ids for text portion
        labels = torch.full((B, T + L), -100, device=z.device, dtype=torch.long)
        labels[:, -L:] = target_ids

        # Attention mask
        attention_mask = torch.ones(B, T + L, device=z.device, dtype=torch.long)

        # =====================================================================
        # Forward through Moshi decoder for text loss
        # =====================================================================
        outputs = self.model.decoder(
            inputs_embeds=full_embeds.to(self._dtype),
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs.loss

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
        Generate text from audio using Moshi's standard generation pipeline.

        Args:
            wav: Audio waveform [1, T] or [B, 1, T] at 24kHz
            max_tokens: Maximum text tokens to generate
            prompt: Unused (Moshi doesn't use text prompts for audio understanding)
            max_new_tokens: Alias for max_tokens

        Returns:
            Generated text string
        """
        if max_new_tokens is not None:
            max_tokens = max_new_tokens

        with torch.no_grad():
            # Ensure shape is [B, 1, T]
            if wav.dim() == 2:
                wav = wav.unsqueeze(0)  # [1, T] → [1, 1, T]
            elif wav.dim() == 1:
                wav = wav.unsqueeze(0).unsqueeze(0)

            wav = wav.to(self._device, dtype=self._dtype)

            # Use Moshi's generate with user audio input
            outputs = self.model.generate(
                user_input_values=wav,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
            )

            # Decode text tokens
            text_ids = outputs.sequences  # [B, seq_len]
            # Filter out special tokens
            text = self.tokenizer.batch_decode(
                text_ids, skip_special_tokens=True
            )[0]

        return text.strip()

    def compute_loss(
        self,
        wav: torch.Tensor,
        target_text: str,
        prompt: str = None,
    ) -> torch.Tensor:
        """
        Compute loss from raw audio (BaseAudioModel interface).

        Encodes wav through Mimi encoder to get VQ latents, then delegates
        to compute_loss_from_latents for the soft-RVQ differentiable path.

        Note: For the latent attack loop, prefer calling compute_loss_from_latents
        directly to avoid redundant encode-decode cycles.

        Args:
            wav: Audio waveform [1, T] or [B, 1, T] at 24kHz
            target_text: Target text
            prompt: Unused

        Returns:
            Loss tensor (scalar, with grad_fn)
        """
        # Ensure [B, 1, T]
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        wav = wav.to(self._device, dtype=self._dtype)

        # Encode through Mimi encoder → encoder_transformer → downsample
        embeddings = self.mimi.encoder(wav)
        encoder_out = self.mimi.encoder_transformer(embeddings.transpose(1, 2))
        embeddings = encoder_out[0].transpose(1, 2)
        embeddings = self.mimi.downsample(embeddings)

        return self.compute_loss_from_latents(embeddings, target_text)
