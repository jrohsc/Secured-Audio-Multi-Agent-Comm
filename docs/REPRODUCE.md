# Reproducing the paper's results

This document describes how to set up the environment, obtain the required model checkpoints and music carriers, and re-run each table.

## 1. Environment

We use conda. Create three environments to match the three target audio LLMs (the per-env transformers/torch versions disagree).

```bash
# Primary env: Qwen2-Audio attacks, MIMI / DAC codec wrappers, EnCodec attacks.
conda create -n codec-attack python=3.10
conda activate codec-attack
conda install -c conda-forge ffmpeg     # required by every codec channel
pip install -r requirements.txt

# Audio Flamingo 3 needs transformers >= 5.0.0.dev0.
conda create -n flamingo3 python=3.10
conda activate flamingo3
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
pip install --upgrade "transformers>=5.0.0.dev0"

# Qwen2.5-Omni-3B / 7B.
conda create -n qwen-omni python=3.10
conda activate qwen-omni
conda install -c conda-forge ffmpeg
pip install -r requirements_omni.txt
```

## 1b. Verify the install

Run the smoke check in whichever environment you just built. It compiles every
source file, resolves every import, checks the derived data paths, and locates
`ffmpeg` — without downloading model weights or running an attack:

```bash
python scripts/check_env.py
```

It exits 0 when everything required is present. Missing optional pieces (the
copyrighted carriers, Kimi-Audio, AudioSeal) are reported as `WARN` and do not
fail the check.

## 2. Model checkpoints (set these before running)

These are **optional**. Each model path falls back to its public HuggingFace repo
id, so the scripts work with no configuration (weights download on first use).
Set the variables only to point at a local snapshot:

```bash
export MODEL_PATH_QWEN2_AUDIO=/path/to/Qwen2-Audio-7B-Instruct
export MODEL_PATH_QWEN25_OMNI=/path/to/Qwen2.5-Omni-7B
export MODEL_PATH_QWEN25_OMNI_3B=/path/to/Qwen2.5-Omni-3B
export MODEL_PATH_AUDIO_FLAMINGO_3=/path/to/audio-flamingo-3-hf
export MODEL_PATH_KIMI_AUDIO=/path/to/Kimi-Audio-7B-Instruct   # held-out victim
export MODEL_PATH_JUDGE_LLM=/path/to/Llama-3.2-3B-Instruct     # LLM judge
```

The variable → fallback mapping lives in `MODEL_PATH_ENV` in `scripts/config.py`.
Each model is fetched from HuggingFace under its respective license:
- `Qwen/Qwen2-Audio-7B-Instruct`
- `Qwen/Qwen2.5-Omni-7B`, `Qwen/Qwen2.5-Omni-3B`
- `nvidia/audio-flamingo-3-hf`
- `moonshotai/Kimi-Audio-7B-Instruct` (needs its own `kimi-audio` conda env, and
  `KIMI_INFER_ROOT` pointing at a checkout of the Kimi-Audio repo for `kimia_infer`)

Two further paths are configurable:

```bash
export FFMPEG_BIN=/path/to/ffmpeg   # if ffmpeg is not on PATH
export BUNDLE_DIR=/path/to/bundle   # score a specific run with eval_quality_*.py
```

## 3. Carriers and target speech

- Copyrighted music carriers (9 files): obtain through legitimate means and place at `data/music/copyrighted/<name>.mp3` (see `docs/MANIFEST_AUDIO.md` for SHA-256 hashes).
- Target speech (TTS-synthesized) is already included in `data/speech/`. To regenerate it from the `.txt` scripts under `data/speech/<scenario>/scripts/`, run the per-scenario `generate.py` (uses Microsoft Edge TTS):
  ```bash
  cd data/speech/01_finance         && python generate.py
  cd data/speech/02_interview/en    && python generate.py
  cd data/speech/04_podcast_advert  && python generate.py
  ```
- `scripts/build_target_maps.py` builds per-scenario `_target_map.json` files by tracing each carrier WAV back to the matching target string in the source configs. Run it from the repo root after carriers and target speech are in place.

## 4. Running an attack

Cross-codec (S3b, Qwen2.5-Omni):

```bash
# EnCodec eps=1.0
python scripts/run_robust_benchmark.py --scenario s3b --target qwen25_omni --eps 1.0 --steps 1000

# Mimi eps=0.2
python scripts/run_mimi_full.py --mode full

# DAC eps=0.6194
python scripts/run_dac_full.py --mode full
```

Each writes attacked WAVs under `results/<scenario>/<cell>/` along with per-pair `_summary.json` and `_codecs.json` files, then runs the channel-routing + eval phases and writes `rollup.json`. Skip-existing is built in.

## 5. Quality eval

```bash
python scripts/eval_quality_encodec.py    # writes quality.jsonl + quality_rollup.json
python scripts/eval_quality_mimi.py
python scripts/eval_quality_dac.py
```

## 6. Regenerating the LaTeX tables

The paper sources are not included in this repository (the PDF is submitted separately through the conference review system), but the table-building scripts will regenerate the LaTeX fragments from the raw JSON rollups so that reviewers can verify every number:

```bash
cd <repo root>
python scripts/build_main_results_tables.py    # emits tab_S{1,2,3}_*.tex fragments
python scripts/build_s1_codec_latex.py
python scripts/build_cross_model_table.py
python scripts/build_defense_table.py
```

Each script writes a `.tex` fragment to stdout (or a target path it prints) that matches the row in the corresponding paper table.

## 7. Eval data layout in this repository

```
data/eval_rollups/
    01_finance_voice_agent/<model>/eps_*/{rollup,results,summaries}.json
    02_interview_screening/<model>/eps_*/...
    03a_ai_detection_bypass/<model>/eps_*/...
    03b_copyright_bypass/<model>/eps_*/audio/*_summary.json   # per-pair
                                  /audio/*_codecs.json        # per-pair MP3/AAC
                                  /quality/quality.jsonl      # per-cell PESQ/LSD/SNR
                                  /quality/quality_rollup.json
        encodec_eps1.0/, mimi_eps0.2/, dac_eps0.6194/         # cross-codec subset
    04_simulated_ota_physical_agent/...
    _results_watermark/      # AudioSeal defense results
    _results_waveguard/      # WaveGuard defense results
    _results_waveform/       # waveform-PGD baseline
```

Every per-pair `_summary.json` contains the target text, the model's output across opus 16/24/32/64/128/192k and (in `_codecs.json`) MP3 and AAC-LC. Every `quality.jsonl` row is one (pair, channel) tuple with its measured SNR, LSD, PESQ-WB, and dLUFS. These are the raw tables the paper aggregates.
