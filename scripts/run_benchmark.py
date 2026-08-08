"""
Full-scale benchmark: latent-space attacks across all music carriers and
physical agent commands.

For each (music, command) pair:
  1. Run latent-space attack via LatentCodecAttacker
  2. Evaluate in 3 modes:
     a) Direct  - adversarial WAV fed straight to Qwen2-Audio
     b) Opus 64 kbps  - compressed via Opus then evaluated
     c) Opus 128 kbps - compressed via Opus then evaluated

Usage:
    python run_benchmark.py                             # Full run (8 music x 35 commands)
    python run_benchmark.py --music jazz_1              # Single music carrier
    python run_benchmark.py --category navigation       # Commands from one category
    python run_benchmark.py --quick                     # 1 music x 2 commands, 150 steps
    python run_benchmark.py --resume results/benchmark_20260206_120000  # Resume interrupted run
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch

from config import (
    MUSIC_FILES, DEFAULT_MUSIC, AGENT_COMMANDS, RESULTS_DIR,
    LATENT_EPS, LATENT_ALPHA, ATTACK_STEPS, PERCEPTUAL_WEIGHT,
    OPUS_EVAL_BITRATES, PROMPT_MODES, DEFAULT_PROMPT_MODE, TARGET_MODEL,
    DATASET_CHOICES, load_saycan_commands,
    compute_wer, save_results_summary, ComplianceJudge,
    ENSEMBLE_MODELS, ENSEMBLE_WEIGHTS, ENSEMBLE_WARMUP_STEPS,
    ENSEMBLE_GRAD_NORMALIZE,
)
from music_carrier import load_music_by_name, resolve_music_path
from latent_attack import LatentCodecAttacker


def _music_short_name(name: str) -> str:
    """Derive a clean short name from a music name or file path."""
    if os.path.sep in name or name.endswith(('.mp3', '.wav', '.flac', '.ogg')):
        return os.path.splitext(os.path.basename(name))[0]
    return name

# Map short category prefixes to human-readable names
CATEGORY_PREFIXES = {
    "nav_": "navigation",
    "manip_": "manipulation",
    "loco_": "locomotion",
    "sense_": "sensing",
    "comm_": "communication",
    "safe_": "safety",
    "deliv_": "delivery",
    "insp_": "inspection",
    "maint_": "maintenance",
    "interact_": "interaction",
}


def get_category(cmd_name: str) -> str:
    """Get human-readable category for a command name."""
    for prefix, cat in CATEGORY_PREFIXES.items():
        if cmd_name.startswith(prefix):
            return cat
    return "other"


def filter_commands(category: Optional[str] = None, mode: str = "transcribe",
                    dataset: str = "ours") -> Dict[str, dict]:
    """Filter commands by category prefix and resolve text for the given mode.

    Args:
        category: Filter by category prefix (only for 'ours' dataset)
        mode: 'transcribe' or 'qa'
        dataset: 'ours' for AGENT_COMMANDS, 'saycan' for SayCan dataset

    Returns a flat dict: {cmd_name: {"command": str, "target": str}}
      - "command": the imperative command text (always the transcribe form)
      - "target": what the attack optimizes the model to output (mode-dependent)
        - transcribe mode: same as command (model transcribes what it hears)
        - qa mode: compliance response ("Sure, I will ...")
    """
    if dataset == "saycan":
        cmds = load_saycan_commands()
    else:
        cmds = AGENT_COMMANDS
    if category is not None:
        target_prefix = None
        for prefix, cat in CATEGORY_PREFIXES.items():
            if cat == category.lower():
                target_prefix = prefix
                break
        if target_prefix is None:
            available = ", ".join(sorted(set(CATEGORY_PREFIXES.values())))
            raise ValueError(f"Unknown category '{category}'. Available: {available}")
        cmds = {k: v for k, v in cmds.items() if k.startswith(target_prefix)}

    # Resolve per-mode text
    resolved = {}
    for k, v in cmds.items():
        if isinstance(v, dict):
            command_text = v.get("transcribe", str(v))
            target_text = v.get(mode, command_text)
        else:
            command_text = v
            target_text = v
        resolved[k] = {"command": command_text, "target": target_text}
    return resolved


def load_completed(resume_dir: str, model_subdir: str = "qwen2_audio") -> set:
    """Load already-completed (music, command) pairs from a previous run.

    Checks the model-specific subdir first (new layout), then falls back to
    the old flat layout for backward compatibility.
    """
    completed = set()

    # New layout: {resume_dir}/{model_subdir}/{exp_key}/record.json
    model_dir = os.path.join(resume_dir, model_subdir)
    if os.path.isdir(model_dir):
        for entry in os.listdir(model_dir):
            rec_path = os.path.join(model_dir, entry, "record.json")
            if os.path.isfile(rec_path):
                try:
                    with open(rec_path) as f:
                        rec = json.load(f)
                    completed.add((rec["music"], rec["command_name"]))
                except (json.JSONDecodeError, KeyError):
                    pass

    # Fallback: old flat layout {resume_dir}/{exp_key}/record.json
    if not completed and os.path.isdir(resume_dir):
        for entry in os.listdir(resume_dir):
            rec_path = os.path.join(resume_dir, entry, "record.json")
            if os.path.isfile(rec_path):
                try:
                    with open(rec_path) as f:
                        rec = json.load(f)
                    completed.add((rec["music"], rec["command_name"]))
                except (json.JSONDecodeError, KeyError):
                    pass

    # Fallback to summary.json
    if not completed:
        summary_path = os.path.join(resume_dir, model_subdir, "summary.json")
        if not os.path.exists(summary_path):
            summary_path = os.path.join(resume_dir, "summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    data = json.load(f)
                completed = {(r["music"], r["command_name"]) for r in data.get("results", [])}
            except (json.JSONDecodeError, KeyError):
                pass

    return completed


def eval_opus(attacker: LatentCodecAttacker, result, bitrates: List[int],
              prompt: str = None) -> Dict[int, dict]:
    """Evaluate adversarial audio after Opus compression at given bitrates."""
    return attacker.test_opus_robustness(result, bitrates=bitrates, prompt=prompt)


def run_benchmark(args):
    """Run the full benchmark."""
    # Determine music files (comma-separated support)
    # music_load_keys: original values passed to load_music_by_name (name or path)
    # music_names: clean short names used for directories, records, and display
    if args.music:
        music_load_keys = [m.strip() for m in args.music.split(",")]
        for m in music_load_keys:
            if resolve_music_path(m) is None:
                available = ", ".join(MUSIC_FILES.keys())
                raise ValueError(f"Unknown music '{m}'. Not a known name ({available}) and not a valid file path.")
        music_names = [_music_short_name(m) for m in music_load_keys]
    else:
        music_load_keys = list(MUSIC_FILES.keys())
        music_names = list(MUSIC_FILES.keys())

    # Determine commands (resolve text for the active prompt mode)
    commands = filter_commands(args.category, mode=args.prompt_mode, dataset=args.dataset)
    if args.commands:
        # Subset by explicit names
        commands = {k: v for k, v in commands.items() if k in args.commands}

    opus_bitrates = [int(b) for b in args.opus_bitrates.split(",")]

    # Resolve prompt mode
    prompt_mode = args.prompt_mode
    if prompt_mode not in PROMPT_MODES:
        available = ", ".join(PROMPT_MODES.keys())
        raise ValueError(f"Unknown prompt mode '{prompt_mode}'. Available: {available}")
    prompt_text = PROMPT_MODES[prompt_mode]["prompt"]

    print(f"Benchmark: {len(music_names)} music files x {len(commands)} commands "
          f"= {len(music_names) * len(commands)} experiments")
    print(f"Dataset: {args.dataset} ({len(commands)} commands)")
    print(f"Attack mode: {'UNTARGETED (disrupt)' if args.untargeted else 'targeted'}")
    print(f"Prompt mode: {prompt_mode} — \"{prompt_text}\"")
    print(f"Opus evaluation bitrates: {opus_bitrates} kbps")
    print(f"Attack steps: {args.steps}, eps: {args.eps}, alpha: {args.alpha}")

    # Output directory (--output-dir for shared parallel dir, --resume to continue)
    if args.output_dir:
        output_dir = args.output_dir
    elif args.resume:
        output_dir = args.resume
    else:
        # Name from config: benchmark_{target}_{mode}_eps{X}_{music}
        music_tag = "_".join(music_names) if len(music_names) <= 3 else f"{len(music_names)}music"
        dataset_tag = f"_{args.dataset}" if args.dataset != "ours" else ""
        attack_mode = "untargeted" if args.untargeted else prompt_mode
        if getattr(args, 'ensemble', False):
            # Seeded runs get their own bundle so the original (zeros-init,
            # unseeded) bundle is never overwritten or resumed into.
            seed_tag = f"_seed{args.seed}" if getattr(args, "rand_init", False) else ""
            dir_name = (f"benchmark_ensemble_{attack_mode}_eps{args.eps}"
                        f"_{music_tag}{dataset_tag}{seed_tag}")
        else:
            dir_name = f"benchmark_{args.target_model}_{attack_mode}_eps{args.eps}_{music_tag}{dataset_tag}"
        output_dir = os.path.join(RESULTS_DIR, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Subdirectories for new layout
    audio_dir = os.path.join(output_dir, "audio")
    model_subdir = args.target_model
    model_dir = os.path.join(output_dir, model_subdir)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Load completed pairs (from per-experiment record files — safe for parallel)
    completed = load_completed(output_dir, model_subdir=model_subdir)
    if completed:
        print(f"Skipping {len(completed)} already-completed experiments")

    # Initialize attacker (load model + EnCodec once)
    print("\nInitializing attacker...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if getattr(args, 'ensemble', False):
        from ensemble_attack import EnsembleLatentAttacker

        # Parse ensemble weights
        ensemble_weights = None
        if args.ensemble_weights:
            ensemble_weights = {}
            for pair in args.ensemble_weights.split(","):
                model_name, weight = pair.split(":")
                ensemble_weights[model_name.strip()] = float(weight.strip())

        ensemble_models = None
        if args.ensemble_models:
            ensemble_models = [m.strip() for m in args.ensemble_models.split(",")]

        attacker = EnsembleLatentAttacker(
            ensemble_models=ensemble_models,
            ensemble_weights=ensemble_weights,
            primary_model=args.target_model,
            eps=args.eps,
            alpha=args.alpha,
            perceptual_weight=args.perceptual_weight,
            warmup_steps=args.ensemble_warmup,
            grad_normalize=not args.no_grad_normalize,
            device=device,
            verbose=not args.quiet,
            rand_init=getattr(args, "rand_init", False),
            seed=getattr(args, "seed", 1),
        )
    else:
        attacker = LatentCodecAttacker(
            target_model=args.target_model,
            eps=args.eps,
            alpha=args.alpha,
            perceptual_weight=args.perceptual_weight,
            device=device,
            verbose=not args.quiet,
        )

    # Save experiment config
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        config_data = {
            "target_model": args.target_model,
            "dataset": args.dataset,
            "music_files": music_names,
            "commands": {k: v["command"] for k, v in commands.items()},
            "targets": {k: v["target"] for k, v in commands.items()},
            "opus_bitrates": opus_bitrates,
            "attack_steps": args.steps,
            "eps": args.eps,
            "alpha": args.alpha,
            "perceptual_weight": args.perceptual_weight,
            "prompt_mode": prompt_mode,
            "prompt_text": prompt_text,
            "untargeted": args.untargeted,
        }
        if getattr(args, 'ensemble', False):
            config_data["ensemble"] = True
            config_data["ensemble_models"] = (
                [m.strip() for m in args.ensemble_models.split(",")]
                if args.ensemble_models else ENSEMBLE_MODELS
            )
            config_data["ensemble_warmup"] = args.ensemble_warmup
            config_data["grad_normalize"] = not args.no_grad_normalize
            # Recorded so a bundle is self-describing: without rand_init the
            # attack is deterministic and `seed` has no effect on the result.
            config_data["rand_init"] = getattr(args, "rand_init", False)
            config_data["seed"] = getattr(args, "seed", 1)
            if args.ensemble_weights:
                config_data["ensemble_weights"] = args.ensemble_weights
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    # Run experiments
    all_results = []

    # Load existing results from model-specific dir (new layout)
    summary_path = os.path.join(model_dir, "summary.json")
    if os.path.isdir(model_dir):
        for entry in sorted(os.listdir(model_dir)):
            rec_path = os.path.join(model_dir, entry, "record.json")
            if os.path.isfile(rec_path):
                try:
                    with open(rec_path) as f:
                        all_results.append(json.load(f))
                except (json.JSONDecodeError, KeyError):
                    pass

    # Fallback: old flat layout
    if not all_results:
        for entry in sorted(os.listdir(output_dir)):
            rec_path = os.path.join(output_dir, entry, "record.json")
            if os.path.isfile(rec_path):
                try:
                    with open(rec_path) as f:
                        all_results.append(json.load(f))
                except (json.JSONDecodeError, KeyError):
                    pass

    total = len(music_names) * len(commands)
    done = len(completed)

    for mi, music_name in enumerate(music_names):
        # Load music once per carrier (use load key which may be a path)
        music_wav = load_music_by_name(music_load_keys[mi])

        for ci, (cmd_name, cmd_info) in enumerate(commands.items()):
            cmd_text = cmd_info["command"]   # imperative command description
            target_text = cmd_info["target"] # what the model should output
            exp_idx = mi * len(commands) + ci + 1

            if (music_name, cmd_name) in completed:
                continue

            done += 1
            print(f"\n{'=' * 70}")
            print(f"[{done}/{total}] music={music_name}, cmd={cmd_name}")
            print(f"Command: \"{cmd_text}\"")
            print(f"Target:  \"{target_text}\"")
            print(f"{'=' * 70}")

            start_time = time.time()

            # Run attack (includes direct evaluation)
            result = attacker.attack(
                music_wav,
                target_text=target_text,
                steps=args.steps,
                music_name=music_name,
                prompt=prompt_text,
                untargeted=args.untargeted,
            )

            # Opus evaluation
            opus_results = eval_opus(attacker, result, opus_bitrates, prompt=prompt_text)

            duration = time.time() - start_time

            # Build result record
            # For untargeted, target_text is the original output (what we're disrupting)
            effective_target = result.target_text if args.untargeted else target_text
            record = {
                "music": music_name,
                "command_name": cmd_name,
                "command_text": cmd_text,
                "target_text": effective_target,
                "category": get_category(cmd_name),
                # Direct evaluation
                "direct": {
                    "success": result.success,
                    "output": result.adversarial_output,
                    "snr_db": result.snr_db,
                    "latent_snr_db": result.latent_snr_db,
                    "final_loss": result.final_loss,
                },
                # Opus evaluations
                "opus": {},
                # Metadata
                "prompt_mode": prompt_mode,
                "prompt_text": prompt_text,
                "untargeted": args.untargeted,
                "steps": result.steps_taken,
                "duration_s": duration,
                "original_output": result.original_output,
            }

            for bw, opus_r in opus_results.items():
                # For untargeted: check WER between original output and
                # Opus-compressed adversarial output
                if args.untargeted:
                    opus_wer_vs_orig = compute_wer(result.original_output, opus_r["output"])
                    opus_success = opus_wer_vs_orig > 0.5
                else:
                    opus_success = opus_r["matches_target"]
                opus_entry = {
                    "success": opus_success,
                    "output": opus_r["output"],
                    "snr_db": opus_r["snr_db"],
                }
                if args.untargeted:
                    opus_entry["wer_vs_original"] = opus_wer_vs_orig
                if prompt_mode == "transcribe" and effective_target:
                    opus_entry["wer"] = compute_wer(effective_target, opus_r["output"])
                record["opus"][str(bw)] = opus_entry

            # Compute WER
            if args.untargeted:
                record["direct"]["wer_vs_original"] = compute_wer(
                    result.original_output, result.adversarial_output
                )
            if prompt_mode == "transcribe" and effective_target:
                record["direct"]["wer"] = compute_wer(effective_target, result.adversarial_output)

            all_results.append(record)

            # Save adversarial WAV to shared audio/ dir (once, skip if exists)
            exp_key = f"{music_name}__{cmd_name}"
            attacker.save_adversarial_wav(result, audio_dir, exp_key)

            # Save record.json under model-specific dir
            exp_record_dir = os.path.join(model_dir, exp_key)
            os.makedirs(exp_record_dir, exist_ok=True)
            rec_path = os.path.join(exp_record_dir, "record.json")
            with open(rec_path, "w") as f:
                json.dump(record, f, indent=2)

            # Incrementally save summary (crash-safe)
            _save_summary(summary_path, all_results, music_names, commands, opus_bitrates)
            save_results_summary(output_dir, all_results, total=total)

            # Quick progress line
            d_ok = "OK" if result.success else "FAIL"
            opus_status = " | ".join(
                f"opus{bw}={'OK' if opus_r['matches_target'] else 'FAIL'}"
                for bw, opus_r in opus_results.items()
            )
            print(f"  Result: direct={d_ok} | {opus_status} | SNR={result.snr_db:.1f}dB | {duration:.1f}s")

    # Final summary
    _save_summary(summary_path, all_results, music_names, commands, opus_bitrates)
    _print_summary(all_results, music_names, commands, opus_bitrates)

    print(f"\nResults saved to: {output_dir}")


def _save_summary(path: str, results: list, music_names: list,
                  commands: dict, opus_bitrates: list):
    """Save summary JSON with results and aggregate stats."""
    # Compute aggregate stats
    n = len(results)
    if n == 0:
        stats = {}
    else:
        direct_successes = sum(1 for r in results if r["direct"]["success"])
        stats = {
            "total_experiments": n,
            "direct_success_rate": direct_successes / n,
            "direct_successes": direct_successes,
        }
        for bw in opus_bitrates:
            bw_key = str(bw)
            opus_successes = sum(
                1 for r in results
                if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"]
            )
            stats[f"opus_{bw}_success_rate"] = opus_successes / n
            stats[f"opus_{bw}_successes"] = opus_successes

        # Per-category stats
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "direct_ok": 0}
                for bw in opus_bitrates:
                    categories[cat][f"opus_{bw}_ok"] = 0
            categories[cat]["total"] += 1
            if r["direct"]["success"]:
                categories[cat]["direct_ok"] += 1
            for bw in opus_bitrates:
                bw_key = str(bw)
                if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"]:
                    categories[cat][f"opus_{bw}_ok"] += 1
        stats["by_category"] = categories

        # Per-music stats
        by_music = {}
        for r in results:
            m = r["music"]
            if m not in by_music:
                by_music[m] = {"total": 0, "direct_ok": 0}
                for bw in opus_bitrates:
                    by_music[m][f"opus_{bw}_ok"] = 0
            by_music[m]["total"] += 1
            if r["direct"]["success"]:
                by_music[m]["direct_ok"] += 1
            for bw in opus_bitrates:
                bw_key = str(bw)
                if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"]:
                    by_music[m][f"opus_{bw}_ok"] += 1
        stats["by_music"] = by_music

        # Average WER (transcription mode)
        direct_wers = [r["direct"]["wer"] for r in results if "wer" in r.get("direct", {})]
        if direct_wers:
            stats["direct_avg_wer"] = sum(direct_wers) / len(direct_wers)
        for bw in opus_bitrates:
            bw_key = str(bw)
            opus_wers = [
                r["opus"][bw_key]["wer"] for r in results
                if bw_key in r.get("opus", {}) and "wer" in r["opus"][bw_key]
            ]
            if opus_wers:
                stats[f"opus_{bw}_avg_wer"] = sum(opus_wers) / len(opus_wers)

    summary = {
        "stats": stats,
        "results": results,
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def _print_summary(results: list, music_names: list, commands: dict,
                   opus_bitrates: list):
    """Print a results matrix to stdout."""
    if not results:
        print("No results to summarize.")
        return

    n = len(results)
    direct_ok = sum(1 for r in results if r["direct"]["success"])

    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    print(f"Total experiments: {n}")
    print(f"Direct success rate: {direct_ok}/{n} ({100*direct_ok/n:.1f}%)")

    # Print WER if available
    direct_wers = [r["direct"]["wer"] for r in results if "wer" in r.get("direct", {})]
    if direct_wers:
        print(f"Direct avg WER: {sum(direct_wers)/len(direct_wers):.3f}")

    for bw in opus_bitrates:
        bw_key = str(bw)
        ok = sum(1 for r in results if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"])
        opus_wers = [
            r["opus"][bw_key]["wer"] for r in results
            if bw_key in r.get("opus", {}) and "wer" in r["opus"][bw_key]
        ]
        wer_str = f" | avg WER: {sum(opus_wers)/len(opus_wers):.3f}" if opus_wers else ""
        print(f"Opus {bw}kbps success rate: {ok}/{n} ({100*ok/n:.1f}%){wer_str}")

    # Per-music table
    print(f"\n{'':─<90}")
    print("PER-MUSIC SUCCESS RATES")
    print(f"{'':─<90}")
    header = f"{'Music':<22} {'Direct':>10}"
    for bw in opus_bitrates:
        header += f" {'Opus'+str(bw):>10}"
    header += f" {'Avg SNR':>10}"
    print(header)
    print("-" * 90)

    for m in music_names:
        m_results = [r for r in results if r["music"] == m]
        if not m_results:
            continue
        m_n = len(m_results)
        d_ok = sum(1 for r in m_results if r["direct"]["success"])
        avg_snr = sum(r["direct"]["snr_db"] for r in m_results) / m_n

        line = f"{m:<22} {d_ok:>3}/{m_n:<3} ({100*d_ok/m_n:4.0f}%)"
        for bw in opus_bitrates:
            bw_key = str(bw)
            o_ok = sum(1 for r in m_results if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"])
            line += f" {o_ok:>3}/{m_n:<3} ({100*o_ok/m_n:4.0f}%)"
        line += f" {avg_snr:>8.1f}dB"
        print(line)

    # Per-category table
    print(f"\n{'':─<90}")
    print("PER-CATEGORY SUCCESS RATES")
    print(f"{'':─<90}")
    header = f"{'Category':<22} {'Direct':>10}"
    for bw in opus_bitrates:
        header += f" {'Opus'+str(bw):>10}"
    print(header)
    print("-" * 90)

    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        c_results = [r for r in results if r["category"] == cat]
        c_n = len(c_results)
        d_ok = sum(1 for r in c_results if r["direct"]["success"])
        line = f"{cat:<22} {d_ok:>3}/{c_n:<3} ({100*d_ok/c_n:4.0f}%)"
        for bw in opus_bitrates:
            bw_key = str(bw)
            o_ok = sum(1 for r in c_results if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"])
            line += f" {o_ok:>3}/{c_n:<3} ({100*o_ok/c_n:4.0f}%)"
        print(line)

    # Full matrix: rows=commands, columns=music, cells= D/O64/O128
    print(f"\n{'':─<90}")
    print("FULL RESULTS MATRIX (D=Direct, 6=Opus64, 8=Opus128 | +=success, -=fail)")
    print(f"{'':─<90}")

    # Compact music headers
    short_music = [m.replace("classical_music", "class").replace("christmas_jazz", "xmas") for m in music_names]
    header = f"{'Command':<22}"
    for sm in short_music:
        header += f" {sm:>9}"
    print(header)
    print("-" * (22 + 10 * len(music_names)))

    # Build lookup
    lookup = {}
    for r in results:
        lookup[(r["music"], r["command_name"])] = r

    for cmd_name in commands:
        line = f"{cmd_name:<22}"
        for m in music_names:
            r = lookup.get((m, cmd_name))
            if r is None:
                line += f" {'---':>9}"
            else:
                d = "+" if r["direct"]["success"] else "-"
                parts = [f"D{d}"]
                for bw in opus_bitrates:
                    bw_key = str(bw)
                    if bw_key in r.get("opus", {}):
                        o = "+" if r["opus"][bw_key]["success"] else "-"
                    else:
                        o = "?"
                    if bw == 64:
                        parts.append(f"6{o}")
                    elif bw == 128:
                        parts.append(f"8{o}")
                    else:
                        parts.append(f"{bw}{o}")
                cell = "/".join(parts)
                line += f" {cell:>9}"
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Full-scale benchmark: latent attacks on physical agent commands"
    )

    # Scope
    parser.add_argument("--music", type=str, default=None,
                        help=f"Comma-separated music carriers. Options: {', '.join(MUSIC_FILES.keys())}")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter commands by category (navigation, manipulation, locomotion, sensing, communication, safety)")
    parser.add_argument("--commands", nargs="+", default=None,
                        help="Explicit command names to run")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Shared output dir (for parallel runs writing to the same place)")

    # Attack params
    parser.add_argument("--steps", type=int, default=ATTACK_STEPS)
    parser.add_argument("--eps", type=float, default=LATENT_EPS)
    parser.add_argument("--alpha", type=float, default=LATENT_ALPHA)
    parser.add_argument("--perceptual-weight", type=float, default=PERCEPTUAL_WEIGHT)

    # Evaluation
    parser.add_argument("--opus-bitrates", type=str,
                        default=",".join(str(b) for b in OPUS_EVAL_BITRATES),
                        help="Comma-separated Opus bitrates in kbps (default: 64,128)")

    # Prompt mode
    parser.add_argument("--prompt-mode", type=str, default=DEFAULT_PROMPT_MODE,
                        help=f"Prompt mode: {', '.join(PROMPT_MODES.keys())} (default: {DEFAULT_PROMPT_MODE})")

    # Target model
    parser.add_argument("--target-model", type=str, default=TARGET_MODEL,
                        help=f"Target model to attack (default: {TARGET_MODEL})")

    # Dataset
    parser.add_argument("--dataset", type=str, default="ours",
                        choices=DATASET_CHOICES,
                        help="Command dataset: 'ours' (200 agent commands) or 'saycan' (81 Google SayCan commands)")

    # Attack mode
    parser.add_argument("--untargeted", action="store_true",
                        help="Untargeted attack: disrupt model output instead of forcing specific text")

    # Ensemble attack
    parser.add_argument("--ensemble", action="store_true",
                        help="Use ensemble attack (optimize against multiple models)")
    parser.add_argument("--ensemble-models", type=str, default=None,
                        help="Comma-separated model names for ensemble (default: qwen2_audio,qwen25_omni,audio_flamingo)")
    parser.add_argument("--ensemble-weights", type=str, default=None,
                        help="Model weights as 'model:weight' pairs (e.g. 'qwen2_audio:2.0,qwen25_omni:1.0')")
    parser.add_argument("--ensemble-warmup", type=int, default=ENSEMBLE_WARMUP_STEPS,
                        help=f"Warmup steps on primary model before adding ensemble (default: {ENSEMBLE_WARMUP_STEPS})")
    parser.add_argument("--rand-init", action="store_true",
                        help="Random-init the latent perturbation (ensemble attack). "
                             "OFF by default: the attack is otherwise deterministic, so "
                             "a seed sweep without this produces identical bundles.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed for --rand-init; also tags the output bundle dir.")
    parser.add_argument("--no-grad-normalize", action="store_true",
                        help="Disable per-model gradient normalization")

    # Convenience
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 music, 2 commands, 150 steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to previous benchmark output dir to resume")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce per-step logging")

    args = parser.parse_args()

    if args.quick:
        if not args.music:
            args.music = DEFAULT_MUSIC
        args.steps = 150
        # Pick first 2 commands from the selected dataset
        if args.dataset == "saycan":
            cmd_source = load_saycan_commands()
        else:
            cmd_source = AGENT_COMMANDS
        first_two = list(cmd_source.keys())[:2]
        args.commands = args.commands or first_two
        args.quiet = True

    run_benchmark(args)


if __name__ == "__main__":
    main()
