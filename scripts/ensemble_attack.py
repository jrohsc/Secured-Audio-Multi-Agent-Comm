"""
Ensemble adversarial attack on multiple Audio LLMs via gradient stitching.

Optimizes a single latent-space perturbation against multiple models
simultaneously by accumulating per-model gradients at the audio_16k level,
then backpropagating through the shared EnCodec decoder.

Gradient flow:
    delta (learnable) -> z_adv = z_original + delta
    -> EnCodec.decode(z_adv) -> audio_24kHz -> resample -> audio_16kHz
                                                             |
                                                     (detach + clone)
                                              ┌── Model A: compute_loss -> grad_A
                                              ├── Model B: compute_loss -> grad_B
                                              └── Model C (subprocess): grad_C via file
                                                             |
                                        audio_16k.backward(weighted_sum_of_grads)
                                                             |
                                                     delta.grad updated

Models in the same conda env run in-process; models requiring different envs
(e.g. Audio Flamingo 3 needs transformers 5.x) run via gradient_worker.py
in a subprocess.
"""

import os
import sys
import json
import time
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import numpy as np

from config import (
    CODECATTACK_LIB_ROOT, MODEL_PATHS, TARGET_MODEL,
    ENCODEC_BANDWIDTH, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    LATENT_EPS, LATENT_ALPHA, ATTACK_STEPS, CHECK_EVERY,
    PERCEPTUAL_WEIGHT, compute_wer,
    ENSEMBLE_MODELS, ENSEMBLE_WEIGHTS, ENSEMBLE_IN_PROCESS,
    ENSEMBLE_SUBPROCESS, ENSEMBLE_WARMUP_STEPS, ENSEMBLE_GRAD_NORMALIZE,
    CONDA_ENVS, PROJECT_ROOT,
)
from music_carrier import mel_distance, multi_scale_mel_distance
from latent_attack import AttackResult, LatentCodecAttacker, UNTARGETED_WER_THRESHOLD

sys.path.insert(0, CODECATTACK_LIB_ROOT)


def _load_model(model_name: str, device: str, dtype: torch.dtype):
    """Load a target model by name. Returns the model wrapper instance."""
    model_path = MODEL_PATHS.get(model_name)
    if model_name == "qwen2_audio":
        from models.qwen2_audio import Qwen2AudioModel
        return Qwen2AudioModel(model_path=model_path, device=device, dtype=dtype)
    elif model_name == "audio_flamingo":
        from models.audio_flamingo import AudioFlamingoModel
        return AudioFlamingoModel(model_path=model_path, device=device, dtype=dtype)
    elif model_name == "kimi_audio":
        from models.kimi_audio import KimiAudioModel
        return KimiAudioModel(model_path=model_path, device=device, dtype=dtype)
    elif model_name == "qwen25_omni":
        from models.qwen25_omni import Qwen25OmniModel
        return Qwen25OmniModel(model_path=model_path, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown model: {model_name}")


class EnsembleLatentAttacker:
    """
    Ensemble adversarial attack using gradient stitching across multiple
    Audio LLMs. Produces a single perturbation that fools multiple models.

    Compatible with LatentCodecAttacker interface: returns AttackResult,
    supports test_opus_robustness(), save_adversarial_wav(), etc.
    """

    def __init__(
        self,
        ensemble_models: List[str] = None,
        ensemble_weights: Dict[str, float] = None,
        in_process_models: List[str] = None,
        subprocess_models: List[str] = None,
        primary_model: str = TARGET_MODEL,
        encodec_bandwidth: float = ENCODEC_BANDWIDTH,
        eps: float = LATENT_EPS,
        alpha: float = LATENT_ALPHA,
        perceptual_weight: float = PERCEPTUAL_WEIGHT,
        warmup_steps: int = ENSEMBLE_WARMUP_STEPS,
        grad_normalize: bool = ENSEMBLE_GRAD_NORMALIZE,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        rand_init: bool = False,
        seed: int = 1,
        rand_init_scale: float = 0.01,
        channel_mode: str = "none",
        multi_bitrate_kbps: List[int] = None,
        channel_warmup_ratio: float = 0.3,
    ):
        # Multi-bitrate Opus EoT. The ensemble attacker was originally
        # clean-channel only (its test_opus_robustness is eval-time), which is
        # why the transcribe-mode ensemble run needed eps=10.0 to survive Opus.
        # channel_mode="multi_bitrate" routes the decoded audio through the same
        # DifferentiableOpusProxy (STE/BPDA) that robust_latent_attack.py uses,
        # so an ensemble arm can be run under the paper's codec-hardened
        # conditions. Add-only: channel_mode="none" keeps the original path.
        self.channel_mode = channel_mode
        self.channel_warmup_ratio = float(channel_warmup_ratio)
        self._multi_bitrate_kbps = multi_bitrate_kbps or [16, 24, 32, 64, 128]
        self.opus_proxy = None
        if channel_mode == "multi_bitrate":
            from channel_augmentation import DifferentiableOpusProxy
            self.opus_proxy = DifferentiableOpusProxy(
                sample_rate=ENCODEC_SAMPLE_RATE,
                bitrates_kbps=self._multi_bitrate_kbps,
            )
        elif channel_mode != "none":
            raise ValueError(f"unsupported channel_mode: {channel_mode}")

        self.eps = eps
        self.alpha = alpha
        self.perceptual_weight = perceptual_weight
        self.rand_init = rand_init
        self.seed = seed
        self.rand_init_scale = rand_init_scale
        self.warmup_steps = warmup_steps
        self.grad_normalize = grad_normalize
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.primary_model = primary_model

        # Resolve model lists
        self.ensemble_models = ensemble_models or ENSEMBLE_MODELS
        self.ensemble_weights = ensemble_weights or ENSEMBLE_WEIGHTS
        self.in_process_models = in_process_models if in_process_models is not None else ENSEMBLE_IN_PROCESS
        self.subprocess_models = subprocess_models if subprocess_models is not None else ENSEMBLE_SUBPROCESS

        # Filter to only requested models
        self.in_process_models = [m for m in self.in_process_models if m in self.ensemble_models]
        self.subprocess_models = [m for m in self.subprocess_models if m in self.ensemble_models]

        # Normalize weights
        active_models = self.in_process_models + self.subprocess_models
        total_weight = sum(self.ensemble_weights.get(m, 1.0) for m in active_models)
        self.norm_weights = {
            m: self.ensemble_weights.get(m, 1.0) / total_weight
            for m in active_models
        }

        self.log(f"Ensemble models: {active_models}")
        self.log(f"  In-process: {self.in_process_models}")
        self.log(f"  Subprocess: {self.subprocess_models}")
        self.log(f"  Weights: {self.norm_weights}")

        # Pre-warm out-of-env workers BEFORE anything else touches the GPU.
        # PyTorch's caching allocator does not give memory back, so once the
        # in-process victims have run a backward pass the parent can be holding
        # ~38 GB and a worker started lazily at the first ensemble step OOMs
        # during weight load. Claiming the worker's memory first makes startup
        # deterministic instead of dependent on allocator state.
        # (Better still: give the worker its own card via ENSEMBLE_WORKER_GPU.)
        if os.environ.get("ENSEMBLE_PERSISTENT_WORKER", "0") == "1":
            for _m in self.subprocess_models:
                if self._get_persistent_worker(_m) is None:
                    raise RuntimeError(
                        f"persistent worker for {_m} failed to start; refusing to "
                        f"run a degraded ensemble silently. See the [stderr] lines "
                        f"above (usually CUDA OOM -> use ENSEMBLE_WORKER_GPU with "
                        f"--gres=gpu:2, or a larger card)."
                    )

        # Initialize EnCodec
        self._init_encodec(encodec_bandwidth)

        # Load in-process models
        self.models: Dict[str, object] = {}
        for model_name in self.in_process_models:
            self.log(f"Loading in-process model: {model_name}...")
            self.models[model_name] = _load_model(model_name, device, dtype)
            # Gradient checkpointing (opt-in via ENSEMBLE_GRAD_CKPT=1): two 7B audio
            # LLMs + a backward pass exceed a 40GB GPU. Checkpointing trades compute
            # for activation memory so the ensemble fits on smaller cards. On an
            # 80GB A100 it is unnecessary (and slower), so it defaults OFF. The attack
            # differentiates w.r.t. the input (not params), so use_reentrant=False +
            # train() (needed for HF to actually checkpoint) is safe here.
            hf = getattr(self.models[model_name], "model", None)
            if hf is not None and os.environ.get("ENSEMBLE_GRAD_CKPT", "0") == "1":
                try:
                    if hasattr(hf, "config"):
                        hf.config.use_cache = False
                    hf.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                    hf.train()
                    # Zero all dropout so train() mode stays deterministic for the attack.
                    for _m in hf.modules():
                        if isinstance(_m, torch.nn.Dropout):
                            _m.p = 0.0
                    self.log(f"  Gradient checkpointing ENABLED for {model_name}")
                except Exception as e:
                    self.log(f"  WARN: grad-checkpoint enable failed for {model_name}: {e}")
            self.log(f"  Loaded {model_name}")

        # Primary model reference (for generate() during attack checks)
        if primary_model in self.models:
            self.target_model = self.models[primary_model]
        elif self.in_process_models:
            self.target_model = self.models[self.in_process_models[0]]
        else:
            raise ValueError("At least one in-process model is required")

        # Gradient worker script path
        self._worker_script = os.path.join(PROJECT_ROOT, "gradient_worker.py")

    def _init_encodec(self, bandwidth: float):
        """Initialize EnCodec model."""
        from encodec_wrapper import EnCodecWrapper
        self.codec = EnCodecWrapper(bandwidth=bandwidth, device=self.device)
        self.log(f"EnCodec loaded (bandwidth={bandwidth} kbps)")

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def encode_music(self, music_wav: torch.Tensor) -> torch.Tensor:
        """Encode music to EnCodec continuous latent space."""
        music_wav = music_wav.to(self.device)
        with torch.no_grad():
            z = self.codec.encode_to_continuous(music_wav)
        self.log(f"Encoded to latent space: {z.shape}")
        return z

    # ---- persistent subprocess workers -------------------------------------
    # The original _compute_subprocess_gradient below spawns a process and
    # RELOADS the model every step (AF3 load ~46 s => ~600 GPU-h for a full
    # ensemble run). These helpers keep one worker alive per out-of-env model.
    # Opt in with ENSEMBLE_PERSISTENT_WORKER=1; the per-step path is untouched
    # so existing behaviour is byte-identical when the flag is off.

    def _get_persistent_worker(self, model_name: str):
        """Spawn (once) and return a live gradient worker for model_name."""
        if not hasattr(self, "_workers"):
            self._workers = {}
        if model_name in self._workers:
            return self._workers[model_name]

        conda_env = CONDA_ENVS.get(model_name)
        if conda_env is None:
            self.log(f"  WARNING: No conda env for {model_name}")
            return None

        conda_base = os.environ.get("CONDA_EXE", "")
        conda_base = (os.path.dirname(os.path.dirname(conda_base))
                      if conda_base else os.path.expanduser("~/miniconda3"))
        activate = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
        script = os.path.join(PROJECT_ROOT, "gradient_worker_persistent.py")

        # A dedicated GPU for the worker if one was provided, else share.
        env = {**os.environ}
        worker_gpu = os.environ.get("ENSEMBLE_WORKER_GPU")
        if worker_gpu:
            env["CUDA_VISIBLE_DEVICES"] = worker_gpu

        cmd = (f'source "{activate}" && conda activate {conda_env} && '
               f'python -u "{script}" --model {model_name}')
        # Worker stderr goes to a FILE, never DEVNULL: this is the only place the
        # reason for a failed start is recorded, and a worker that fails to start
        # silently degrades the run to a smaller ensemble.
        err_path = os.path.join(tempfile.gettempdir(),
                                f"ensemble_worker_{model_name}.err")
        err_fh = open(err_path, "w")
        self._worker_err_paths = getattr(self, "_worker_err_paths", {})
        self._worker_err_paths[model_name] = err_path
        self.log(f"  Starting persistent worker for {model_name} "
                 f"(env={conda_env}, gpu={env.get('CUDA_VISIBLE_DEVICES','inherit')}, "
                 f"stderr={err_path})")
        proc = subprocess.Popen(
            ["bash", "-c", cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=err_fh, text=True, bufsize=1, env=env,
        )
        # Block until the model is resident. A worker that dies during load must
        # fail loudly here rather than silently contributing zero gradients.
        # Skip any banner lines a model wrapper prints to stdout before READY
        # (the worker redirects those to stderr, but don't rely on it).
        handshake = None
        for _ in range(200):
            line = proc.stdout.readline()
            if not line:            # EOF => worker died during load
                break
            line = line.strip()
            if line == "READY":
                handshake = line
                break
            self.log(f"  [worker {model_name}] {line[:120]}")
        if handshake != "READY":
            self.log(f"  ERROR: worker for {model_name} failed to start")
            try:
                err_fh.flush()
                with open(err_path) as fh:
                    tail = fh.read().strip().splitlines()[-15:]
                for ln in tail:
                    self.log(f"    [stderr] {ln[:200]}")
            except Exception:
                pass
            proc.kill()
            return None
        self.log(f"  Worker for {model_name} READY")

        tmpdir = tempfile.mkdtemp(prefix=f"ensw_{model_name}_")
        self._workers[model_name] = (proc, tmpdir)
        return self._workers[model_name]

    def _compute_persistent_gradient(
        self,
        model_name: str,
        audio_16k: torch.Tensor,
        target_text: str,
        prompt: str = None,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """Gradient from a long-lived worker. Returns (None, 0.0) on failure."""
        w = self._get_persistent_worker(model_name)
        if w is None:
            return None, 0.0
        proc, tmpdir = w

        req = os.path.join(tmpdir, "req.pt")
        # The scratch dir lives in the node-local (job-scoped) /tmp and has been
        # observed to disappear mid-run -- job 62193075 died at clip 72/224 with
        # "Parent directory /tmp/ensw_audio_flamingo_* does not exist". Recreate
        # it, and treat a still-failing save as a COUNTED skip (see _subproc_skips
        # in attack()) rather than letting it kill the whole benchmark.
        try:
            os.makedirs(tmpdir, exist_ok=True)
            torch.save({"audio_16k": audio_16k.detach().cpu(),
                        "target_text": target_text,
                        "prompt": prompt or ""}, req)
        except OSError as e:
            self.log(f"  WARNING: worker {model_name} request write failed: {e}")
            self._workers.pop(model_name, None)
            return None, 0.0
        try:
            proc.stdin.write(req + "\n")
            proc.stdin.flush()
            resp = proc.stdout.readline().strip()
        except (BrokenPipeError, ValueError) as e:
            self.log(f"  WARNING: worker {model_name} pipe broke: {e}")
            self._workers.pop(model_name, None)
            return None, 0.0

        if not resp.startswith("OK"):
            self.log(f"  WARNING: worker {model_name} returned {resp!r}")
            return None, 0.0

        out = req + ".out"
        if not os.path.isfile(out):
            self.log(f"  WARNING: worker {model_name} wrote no output file")
            return None, 0.0
        try:
            result = torch.load(out, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError) as e:
            # Same scratch-dir failure mode, on the read side.
            self.log(f"  WARNING: worker {model_name} response read failed: {e}")
            self._workers.pop(model_name, None)
            return None, 0.0
        return result["gradient"].to(self.device), float(result["loss"])

    def close_workers(self):
        """Shut down any persistent workers (safe to call repeatedly)."""
        for name, (proc, tmpdir) in getattr(self, "_workers", {}).items():
            try:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
            self.log(f"  Worker for {name} stopped")
        self._workers = {}

    def _compute_subprocess_gradient(
        self,
        model_name: str,
        audio_16k: torch.Tensor,
        target_text: str,
        prompt: str = None,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """
        Compute gradient for a model running in a separate conda env.

        Saves audio tensor to a temp file, invokes gradient_worker.py in the
        appropriate conda env, and reads back the gradient tensor.

        Returns:
            (gradient, loss_value) or (None, 0.0) on failure
        """
        conda_env = CONDA_ENVS.get(model_name)
        if conda_env is None:
            self.log(f"  WARNING: No conda env for {model_name}, skipping")
            return None, 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "audio_16k.pt")
            output_path = os.path.join(tmpdir, "gradient.pt")

            # Save input audio (detached, on CPU)
            torch.save({
                "audio_16k": audio_16k.detach().cpu(),
                "target_text": target_text,
                "prompt": prompt or "",
                "model_name": model_name,
            }, input_path)

            # Find conda base
            conda_base = os.environ.get("CONDA_EXE", "")
            if conda_base:
                conda_base = os.path.dirname(os.path.dirname(conda_base))
            else:
                conda_base = os.path.expanduser("~/miniconda3")

            activate_script = os.path.join(conda_base, "etc", "profile.d", "conda.sh")

            cmd = (
                f'source "{activate_script}" && '
                f'conda activate {conda_env} && '
                f'python "{self._worker_script}" '
                f'--input "{input_path}" '
                f'--output "{output_path}"'
            )

            try:
                proc = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
                )

                if proc.returncode != 0:
                    self.log(f"  WARNING: Subprocess {model_name} failed (rc={proc.returncode})")
                    if proc.stderr:
                        self.log(f"  stderr: {proc.stderr[:200]}")
                    return None, 0.0

                if not os.path.isfile(output_path):
                    self.log(f"  WARNING: No output file from {model_name}")
                    return None, 0.0

                result = torch.load(output_path, map_location="cpu", weights_only=True)
                grad = result["gradient"].to(self.device)
                loss_val = result["loss"]
                return grad, loss_val

            except subprocess.TimeoutExpired:
                self.log(f"  WARNING: Subprocess {model_name} timed out")
                return None, 0.0
            except Exception as e:
                self.log(f"  WARNING: Subprocess {model_name} error: {e}")
                return None, 0.0

    def _compute_in_process_gradient(
        self,
        model_name: str,
        audio_16k: torch.Tensor,
        target_text: str,
        prompt: str = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute gradient for an in-process model.

        audio_16k must have requires_grad=True (or be a leaf that we can
        get .grad from). We compute loss and call backward to populate
        audio_16k.grad.

        Returns:
            (gradient w.r.t. audio_16k, loss_value)
        """
        model = self.models[model_name]

        # Reset model caches if needed
        if hasattr(model, 'reset_cache'):
            model.reset_cache()

        # Force train() at the call site so HF gradient checkpointing actually
        # activates (the wrapper loads the model in eval(); HF only checkpoints
        # when training=True). Only when checkpointing is opted in; dropout was
        # zeroed at load, so this stays deterministic.
        if os.environ.get("ENSEMBLE_GRAD_CKPT", "0") == "1":
            hf = getattr(model, "model", None)
            if hf is not None:
                hf.train()

        audio_input = audio_16k.detach().requires_grad_(True)
        loss = model.compute_loss(audio_input, target_text, prompt=prompt)
        loss.backward()

        if audio_input.grad is None:
            raise RuntimeError(
                f"No gradient for {model_name}: loss.requires_grad={loss.requires_grad}, "
                f"loss.grad_fn={loss.grad_fn}, audio_input.is_leaf={audio_input.is_leaf}"
            )

        grad = audio_input.grad.detach().clone()
        loss_val = loss.item()

        return grad, loss_val

    def attack(
        self,
        music_wav: torch.Tensor,
        target_text: str,
        steps: int = ATTACK_STEPS,
        check_every: int = CHECK_EVERY,
        music_name: str = "",
        use_multi_scale_loss: bool = False,
        prompt: str = None,
        untargeted: bool = False,
    ) -> AttackResult:
        """
        Run ensemble latent-space adversarial attack.

        Uses gradient stitching: forward through EnCodec decoder once,
        then accumulate per-model gradients independently and chain back
        through the decoder via audio_16k.backward(combined_grad).

        Args:
            music_wav: Music carrier [1, 1, T] at 24kHz
            target_text: Text to force Audio LLMs to output
            steps: Number of optimization steps
            check_every: Check model output every N steps
            music_name: Name of the music file (for logging)
            use_multi_scale_loss: Use multi-scale mel loss
            prompt: Text prompt for the models
            untargeted: If True, maximize loss on original output instead
                of minimizing loss on target_text.

        Returns:
            AttackResult with adversarial audio and metrics
        """
        start_time = time.time()

        music_wav = music_wav.to(self.device)

        # Reset all in-process model caches
        for m in self.models.values():
            if hasattr(m, 'reset_cache'):
                m.reset_cache()

        # Encode to latent space
        z_original = self.encode_music(music_wav)

        # Cache original decoded audio for perceptual loss
        with torch.no_grad():
            original_audio_24k = self.codec.decode_from_continuous(z_original)
            original_audio_16k = torchaudio.functional.resample(
                original_audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            original_output = self.target_model.generate(original_audio_16k, prompt=prompt)

        self.log(f"Original model output: {original_output}")

        # Untargeted mode: disrupt the original output instead of forcing a target
        if untargeted:
            target_text = original_output
            self.log(f"UNTARGETED mode: maximizing loss on original output")
        self.log(f"Target: {target_text}")
        if prompt:
            self.log(f"Prompt: {prompt}")

        # Initialize learnable perturbation.
        # Default stays zeros-init so every previously produced bundle remains
        # reproducible byte-for-byte. Random init is opt-in and exists ONLY to
        # make a seed sweep meaningful: with zeros-init this attack is fully
        # deterministic, so re-running under different SEED values yields
        # identical bundles and a "3-seed" table would be one run reported
        # three times. Mirrors the rand-init in robust_latent_attack.py.
        if self.rand_init:
            gen = torch.Generator(device="cpu").manual_seed(int(self.seed))
            noise = torch.randn(z_original.shape, generator=gen,
                                dtype=torch.float32).to(z_original.device)
            # Scale relative to the latent's own magnitude so the init is a small
            # perturbation regardless of eps units, then clamp into the eps ball.
            noise = noise / noise.norm() * (self.rand_init_scale * z_original.norm())
            noise = noise.clamp(-self.eps, self.eps)
            delta = noise.to(z_original.dtype).clone().requires_grad_(True)
            self.log(f"Random init: seed={self.seed} "
                     f"scale={self.rand_init_scale} |delta_0|={delta.norm().item():.4f}")
        else:
            delta = torch.zeros_like(z_original, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.alpha)

        # History tracking
        history = {
            "loss": [], "behavior_loss": [], "perceptual_loss": [],
            "outputs": [], "steps": [],
            "per_model_loss": {m: [] for m in self.ensemble_models},
        }
        best_delta = delta.data.clone()
        best_loss = float('inf')
        success = False

        all_models = self.in_process_models + self.subprocess_models

        self.log(f"\nStarting ensemble attack: eps={self.eps}, alpha={self.alpha}, steps={steps}")
        self.log(f"Mode: {'untargeted' if untargeted else 'targeted'}")
        self.log(f"Perceptual weight: {self.perceptual_weight}")
        self.log(f"Warmup steps (primary only): {self.warmup_steps}")
        self.log(f"Gradient normalize: {self.grad_normalize}")
        self.log("-" * 60)

        # Enable gradients through EnCodec decoder
        self.codec.model.decoder.train()

        for step in range(steps):
            optimizer.zero_grad()

            # Apply perturbation in latent space
            z_adv = z_original + delta

            # Decode to audio (24kHz) — shared forward pass
            audio_adv_24k = self.codec.decode_from_continuous(z_adv)

            # Multi-bitrate Opus EoT (STE/BPDA). Two-stage schedule mirroring
            # robust_latent_attack: a clean warm-up so the attack first finds the
            # target, then codec-hardened steps for the rest of the budget.
            if self.opus_proxy is not None:
                if step >= int(self.channel_warmup_ratio * steps):
                    audio_adv_24k = self.opus_proxy(audio_adv_24k)

            # Resample to 16kHz — this is the stitching point
            audio_adv_16k = torchaudio.functional.resample(
                audio_adv_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )

            # Decide which models are active this step
            if step < self.warmup_steps:
                active_in_process = [m for m in self.in_process_models if m == self.primary_model]
                active_subprocess = []
            else:
                active_in_process = list(self.in_process_models)
                active_subprocess = list(self.subprocess_models)

            # Compute perceptual loss first with retain_graph=True
            # (model-independent, computed at audio_24k level)
            if self.perceptual_weight > 0:
                if use_multi_scale_loss:
                    p_loss = multi_scale_mel_distance(
                        audio_adv_24k.squeeze(0), original_audio_24k.squeeze(0).detach(),
                        sample_rate=ENCODEC_SAMPLE_RATE
                    )
                else:
                    p_loss = mel_distance(
                        audio_adv_24k.squeeze(0), original_audio_24k.squeeze(0).detach(),
                        sample_rate=ENCODEC_SAMPLE_RATE
                    )
                # Backprop perceptual loss (retain graph for behavior grads)
                (self.perceptual_weight * p_loss).backward(retain_graph=True)
                p_loss_val = p_loss.item()
            else:
                p_loss_val = 0.0

            # --- Gradient stitching ---
            # Collect per-model gradients w.r.t. audio_16k
            combined_grad = torch.zeros_like(audio_adv_16k.detach())
            total_behavior_loss = 0.0
            active_weight_sum = 0.0

            for model_name in active_in_process:
                grad, loss_val = self._compute_in_process_gradient(
                    model_name, audio_adv_16k, target_text, prompt=prompt
                )
                w = self.norm_weights.get(model_name, 1.0)

                if self.grad_normalize:
                    grad_norm = grad.norm()
                    if grad_norm > 1e-8:
                        grad = grad / grad_norm

                combined_grad += w * grad
                total_behavior_loss += w * loss_val
                active_weight_sum += w
                history["per_model_loss"][model_name].append(loss_val)

            for model_name in active_subprocess:
                if os.environ.get("ENSEMBLE_PERSISTENT_WORKER", "0") == "1":
                    grad, loss_val = self._compute_persistent_gradient(
                        model_name, audio_adv_16k, target_text, prompt=prompt
                    )
                else:
                    grad, loss_val = self._compute_subprocess_gradient(
                        model_name, audio_adv_16k, target_text, prompt=prompt
                    )
                if grad is None:
                    # Skip failed subprocess, rescale remaining weights.
                    # NOTE: this is silent-by-design in the original code. Count
                    # the skips so a run where the out-of-env victim contributed
                    # nothing cannot be mistaken for a successful ensemble.
                    self._subproc_skips = getattr(self, "_subproc_skips", 0) + 1
                    continue

                w = self.norm_weights.get(model_name, 1.0)

                if self.grad_normalize:
                    grad_norm = grad.norm()
                    if grad_norm > 1e-8:
                        grad = grad / grad_norm

                combined_grad += w * grad
                total_behavior_loss += w * loss_val
                active_weight_sum += w
                history["per_model_loss"][model_name].append(loss_val)

            # Rescale if some models were skipped
            if active_weight_sum > 0 and active_weight_sum < 1.0:
                combined_grad = combined_grad / active_weight_sum
                total_behavior_loss = total_behavior_loss / active_weight_sum

            # Untargeted: negate gradients to maximize loss (disrupt original output)
            if untargeted:
                combined_grad = -combined_grad
                total_behavior_loss = -total_behavior_loss

            # Backprop combined gradient through the EnCodec decoder
            # audio_adv_16k is connected to z_adv via resample <- decode <- z_adv
            audio_adv_16k.backward(combined_grad)

            optimizer.step()

            # Project to epsilon ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)

            # Track
            total_loss = total_behavior_loss + self.perceptual_weight * p_loss_val
            history["loss"].append(total_loss)
            history["behavior_loss"].append(total_behavior_loss)
            history["perceptual_loss"].append(p_loss_val)

            if total_loss < best_loss:
                best_loss = total_loss
                best_delta = delta.data.clone()

            # Progress logging
            if (step + 1) % 10 == 0:
                model_losses = " | ".join(
                    f"{m}: {history['per_model_loss'][m][-1]:.4f}"
                    for m in all_models
                    if history["per_model_loss"][m]
                )
                phase = "warmup" if step < self.warmup_steps else "ensemble"
                self.log(
                    f"Step {step+1:4d}/{steps} [{phase}] | "
                    f"Total: {total_loss:.4f} | "
                    f"Behavior: {total_behavior_loss:.4f} | "
                    f"Perceptual: {p_loss_val:.4f} | "
                    f"{model_losses}"
                )

            # Check model output periodically (use primary model)
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    z_test = z_original + delta
                    audio_test_24k = self.codec.decode_from_continuous(z_test)
                    audio_test_16k = torchaudio.functional.resample(
                        audio_test_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    output = self.target_model.generate(audio_test_16k, prompt=prompt)

                history["outputs"].append(output)
                history["steps"].append(step + 1)
                self.log(f"  -> Primary output: {output[:100]}")

                if untargeted:
                    wer = compute_wer(target_text, output)
                    if wer > UNTARGETED_WER_THRESHOLD:
                        self.log(f"  -> DISRUPTED at step {step+1}! (WER={wer:.3f} > {UNTARGETED_WER_THRESHOLD})")
                        success = True
                    else:
                        self.log(f"  -> Not disrupted (WER={wer:.3f})")
                else:
                    if target_text.lower() in output.lower():
                        self.log(f"  -> TARGET ACHIEVED at step {step+1}!")
                        success = True

        # Restore eval mode
        self.codec.model.decoder.eval()

        # Final evaluation with best delta (definitive success check)
        with torch.no_grad():
            z_final = z_original + best_delta
            audio_final_24k = self.codec.decode_from_continuous(z_final)
            audio_final_16k = torchaudio.functional.resample(
                audio_final_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            final_output = self.target_model.generate(audio_final_16k, prompt=prompt)

        # Reset success — final output is the definitive answer
        success = False
        if untargeted:
            final_wer = compute_wer(target_text, final_output)
            if final_wer > UNTARGETED_WER_THRESHOLD:
                success = True
            self.log(f"Final WER vs original: {final_wer:.3f} (threshold={UNTARGETED_WER_THRESHOLD})")
        else:
            if target_text.lower() in final_output.lower():
                success = True

        # Compute SNR
        snr_db = LatentCodecAttacker._compute_snr(
            original_audio_24k.squeeze().detach(),
            audio_final_24k.squeeze().detach()
        )
        latent_snr_db = LatentCodecAttacker._compute_snr(
            z_original.detach(), z_final.detach()
        )

        duration = time.time() - start_time

        self.log("\n" + "=" * 60)
        self.log("Ensemble Attack Complete")
        self.log(f"Models: {all_models}")
        self.log(f"Final Loss: {best_loss:.4f}")
        self.log(f"Audio SNR: {snr_db:.2f} dB")
        self.log(f"Latent SNR: {latent_snr_db:.2f} dB")
        self.log(f"Success (primary): {success}")
        self.log(f"Final Output: {final_output}")
        self.log(f"Duration: {duration:.1f}s")
        # Loud on the failure mode that would otherwise look like a clean run:
        # an out-of-env victim whose gradients never made it into the ensemble.
        skips = getattr(self, "_subproc_skips", 0)
        if skips:
            expected = max(0, steps - self.warmup_steps) * len(self.subprocess_models)
            self.log(f"!!! WARNING: {skips}/{expected} subprocess gradient requests "
                     f"FAILED for {self.subprocess_models} — those victims did NOT "
                     f"fully participate in this attack. Do not report it as an "
                     f"{len(self.ensemble_models)}-model ensemble without checking.")
        self.log("=" * 60)

        return AttackResult(
            original_wav=original_audio_24k.detach().cpu().squeeze(0),
            adversarial_wav=audio_final_24k.detach().cpu().squeeze(0),
            original_latents=z_original.detach().cpu(),
            adversarial_latents=z_final.detach().cpu(),
            latent_perturbation=best_delta.detach().cpu(),
            original_output=original_output,
            adversarial_output=final_output,
            target_text=target_text,
            success=success,
            final_loss=best_loss,
            steps_taken=steps,
            snr_db=snr_db,
            latent_snr_db=latent_snr_db,
            history=history,
            music_name=music_name,
            attack_duration_s=duration,
        )

    # ------------------------------------------------------------------
    # Delegate robustness testing and saving to LatentCodecAttacker utils
    # ------------------------------------------------------------------

    def test_opus_robustness(
        self,
        result: AttackResult,
        bitrates: List[int] = None,
        prompt: str = None,
    ) -> Dict[int, dict]:
        """Test Opus robustness using the primary model for evaluation."""
        # Create a lightweight single-model attacker for robustness testing
        # (reuses self.codec and self.target_model)
        helper = object.__new__(LatentCodecAttacker)
        helper.codec = self.codec
        helper.target_model = self.target_model
        helper.device = self.device
        helper.verbose = self.verbose
        helper.log = self.log
        return helper.test_opus_robustness(result, bitrates=bitrates, prompt=prompt)

    def test_encodec_robustness(
        self,
        result: AttackResult,
        bitrates: List[float] = None,
        prompt: str = None,
    ) -> Dict[float, dict]:
        """Test EnCodec re-encoding robustness using the primary model."""
        helper = object.__new__(LatentCodecAttacker)
        helper.codec = self.codec
        helper.target_model = self.target_model
        helper.device = self.device
        helper.verbose = self.verbose
        helper.log = self.log
        helper._bandwidth = getattr(self.codec, '_bandwidth', ENCODEC_BANDWIDTH)
        return helper.test_encodec_robustness(result, bitrates=bitrates, prompt=prompt)

    def save_adversarial_wav(self, result: AttackResult, audio_dir: str,
                             filename: str) -> str:
        """Save adversarial WAV to a shared audio directory."""
        os.makedirs(audio_dir, exist_ok=True)
        wav_path = os.path.join(audio_dir, f"{filename}.wav")
        if os.path.isfile(wav_path):
            self.log(f"Audio already exists, skipping: {wav_path}")
            return wav_path
        import soundfile as sf
        adv_wav = result.adversarial_wav.squeeze().cpu().numpy()
        sf.write(wav_path, adv_wav, ENCODEC_SAMPLE_RATE)
        self.log(f"Saved adversarial audio: {wav_path}")
        return wav_path
