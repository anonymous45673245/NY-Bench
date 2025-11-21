#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_run_inference_layerwisememory.py

Multi-turn image editing inference pipeline for NY-BENCH using the
Layer-wise Memory baseline of Kim et al. [10]
(*Improving Editability in Image Generation with Layer-wise Memory*).

We assume the external repository is placed under:
    <project_root>/models/CVPR/

Inputs:
    - data/images/<base_id>_turn0.png       # e.g., data/images/0001_turn0.png
    - data/scenarios/<session_id>.json      # e.g., data/scenarios/0001_s0.json

Outputs:
    - predictions/LayerwiseMemory/<session_id>_LayerwiseMemory_turn<t>.png  (t = 1, 2, ...)

Example:
    python scripts/01_run_inference_layerwisememory.py \
        --data-root ./data \
        --pred-root ./predictions \
        --gpu-idx 0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw

# -------------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

# External model root (Kim et al. [10] repository)
CVPR_ROOT = ROOT_DIR / "models" / "CVPR"

# Make external repo importable
sys.path.insert(0, str(CVPR_ROOT))
sys.path.insert(0, str(CVPR_ROOT / "diffusers" / "src"))

from diffusers import DPMSolverMultistepScheduler  # type: ignore
from diffusion.sa_solver_diffusers import SASolverScheduler  # type: ignore
from scripts.pipeline_pixart_inpaint_with_latent_memory_improved import (  # type: ignore
    PixArtAlphaInpaintLMPipeline,
)

# -------------------------------------------------------------------------
# Hyper-parameters and memory settings
# -------------------------------------------------------------------------

TURN0_STRENGTH = 0.6      # Strength for the initialization step (Turn 0)
EDIT_STRENGTH = 0.6       # Strength for editing turns (Turn 1, 2, ...)
MAX_LATENT_HISTORY = 2    # Max length of latent memory history
CACHE_CLEAR_INTERVAL = 5  # Clear GPU cache every N scenarios


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def create_mask(resolution: int, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """Create a binary mask from a single bbox (x1, y1, x2, y2)."""
    mask = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)
    return mask


def create_mask_from_bboxes(resolution: int, bboxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    """Create a union mask from multiple bboxes."""
    mask = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(mask)
    for bbox in bboxes:
        draw.rectangle(bbox, fill=255)
    return mask


def get_default_args():
    """Return a default argparse.Namespace for the PixArt pipeline."""
    args = argparse.Namespace()

    args.model_version = "PixArtAlpha"
    args.model_base = "InpaintLM"
    args.batch_size = 1
    args.resolution = 1024

    args.dpms_guidance_scale = 7.5
    args.sas_guidance_scale = 3.0
    args.num_inference_steps = 20
    args.scheduler_type = "DPM-Solver"
    args.GPU_IDX = 0

    args.result_dir = "./output_automation"
    args.exp_name = f"PixArt_Auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Extra hyper-parameters from the original implementation
    args.vanilla_ratio = 0.05
    args.cattn_masking = True
    args.multi_query_disentanglement = True
    args.object_gen = False
    args.utilize_cache = False
    args.remove_vanilla_ratio = 0.5
    args.alpha = 1.0
    args.sigma = 2.0

    # State for latent memory
    args.latent_memory = []
    args.counter = 0

    # Prompt suffix used for editing turns
    args.prompt_suffix = " Do not replace or delete any existing objects in the image."

    # Scheduler / guidance (to be set later)
    args.guidance_scale = None

    # Optional shared generator for reproducibility
    args.generator = None

    return args


def set_scheduler_once(pipe, args):
    """
    Set the scheduler only once per pipeline and return the guidance scale.
    This avoids re-creating the scheduler every call.
    """
    if args.scheduler_type == "DPM-Solver":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        guidance_scale = args.dpms_guidance_scale
    elif args.scheduler_type == "SA-Solver":
        pipe.scheduler = SASolverScheduler.from_config(
            pipe.scheduler.config, algorithm_type="data_prediction"
        )
        guidance_scale = args.sas_guidance_scale
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler_type}")
    return guidance_scale


def run_pipeline(
    pipe,
    args,
    prompt: str,
    image=None,
    mask_image=None,
    inpaint: bool = False,
    cattn_masking: bool = False,
    multi_query_disentanglement: bool = False,
    alpha: float = 1.0,
    object_gen=False,
    utilize_cache=False,
    subject_token_idx: list | None = None,
    sigma: float = 0.0,
    remove_vanilla_ratio: float = 0.5,
    new_generation: bool = True,
    remove_checkbox: bool = False,
    strength: float = 1.0,
    add_suffix: bool = True,
):
    """
    Thin wrapper around PixArtAlphaInpaintLMPipeline.__call__ with latent memory.
    """

    if add_suffix and getattr(args, "prompt_suffix", None):
        prompt = prompt + args.prompt_suffix

    if args.counter > 1:
        latent_memory = args.latent_memory[-1] if args.latent_memory else None
    elif utilize_cache is True:
        latent_memory = args.latent_memory[-1] if args.latent_memory else None
    else:
        latent_memory = None

    if not new_generation and args.latent_memory:
        args.latent_memory.pop(-1)
        latent_memory = args.latent_memory[-1] if args.latent_memory else None

    if remove_checkbox:
        if len(args.latent_memory) >= 3:
            latent_memory = [args.latent_memory[-1], args.latent_memory[-3]]
        elif len(args.latent_memory) > 0:
            print("Warning: Not enough history to remove. Using last latent.")
            latent_memory = args.latent_memory[-1]
        else:
            latent_memory = None

    if args.generator is None:
        args.generator = torch.Generator(device=f"cuda:{args.GPU_IDX}").manual_seed(334)

    pipe_kwargs = {
        "prompt": prompt,
        "image": image,
        "mask_image": mask_image,
        "strength": strength,
        "generator": args.generator,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": args.batch_size,
        "inpaint": inpaint,
        "latent_memory": latent_memory,
        "vanilla_ratio": args.vanilla_ratio,
        "cattn_masking": cattn_masking,
        "subject_token_idx": subject_token_idx,
        "sigma": sigma,
        "remove_vanilla_ratio": remove_vanilla_ratio,
        "multi_query_disentanglement": multi_query_disentanglement,
        "alpha": alpha,
        "object_gen": object_gen,
        "utilize_cache": utilize_cache,
        "new_generation": new_generation,
    }

    if remove_checkbox and len(args.latent_memory) >= 3:
        pipe_kwargs["remove_prev"] = True

    with torch.inference_mode():
        result_tensor, latent_memory_new = pipe(**pipe_kwargs)

    result_image = result_tensor.images[0]

    if not (object_gen is True and utilize_cache is False):
        args.latent_memory.append(latent_memory_new)
        if len(args.latent_memory) > MAX_LATENT_HISTORY:
            args.latent_memory.pop(0)

    return result_image


def clean_memory():
    """Explicitly clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------------------------------------------------------------
# Main LayerwiseMemory pipeline (importable + CLI)
# -------------------------------------------------------------------------

def run_layerwisememory_pipeline(
    scenario_root: Path,
    original_img_root: Path,
    base_results_root: Path,
    gpu_idx: int = 0,
    num_shards: int = 1,
    shard_idx: int = 0,
):
    """
    Run the LayerwiseMemory baseline (Kim et al. [10]) on all scenarios.

    Args:
        scenario_root: Path to data/scenarios/
        original_img_root: Path to data/images/
        base_results_root: Path to predictions/
        gpu_idx: GPU index to use.
        num_shards: Total number of shards (for multi-process evaluation).
        shard_idx: Index of the current shard.
    """
    scenario_files = sorted(f for f in os.listdir(scenario_root) if f.endswith(".json"))

    scenario_dir = scenario_root
    original_dir = original_img_root

    output_root = base_results_root / "LayerwiseMemory"
    output_root.mkdir(parents=True, exist_ok=True)

    args = get_default_args()
    args.GPU_IDX = gpu_idx
    args.result_dir = str(output_root)
    args.exp_name = "LayerwiseMemory_benchmark"

    print("[LayerwiseMemory] Loading PixArtAlpha Inpaint LM pipeline (Kim et al. [10])...")
    pipe_inpaint = PixArtAlphaInpaintLMPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16,
    ).to(f"cuda:{args.GPU_IDX}")
    print("[LayerwiseMemory] Pipeline loaded.")

    args.guidance_scale = set_scheduler_once(pipe_inpaint, args)

    print(f"[LayerwiseMemory] Total scenarios: {len(scenario_files)}")
    print(
        f"[LayerwiseMemory] Sharding: num_shards={num_shards}, "
        f"shard_idx={shard_idx}, gpu_idx={gpu_idx}"
    )

    processed_idx = 0

    for global_idx, scn_name in enumerate(scenario_files):
        if num_shards > 1 and (global_idx % num_shards != shard_idx):
            continue

        processed_idx += 1

        args.latent_memory = []
        args.counter = 0
        args.generator = None

        scn_path = scenario_dir / scn_name
        if not scn_path.exists():
            print(f"[SKIP] Scenario file not found: {scn_path}")
            continue

        with open(scn_path, "r", encoding="utf-8") as f:
            scn = json.load(f)

        session_id = scn.get("session_id", scn_path.stem)
        img_id = session_id.split("_")[0]
        category = scn.get("category", "").upper()

        scenario = scn.get("scenario", {})
        turn_indices = sorted(
            int(k.replace("instruction_turn", "").replace("_natural", ""))
            for k in scenario.keys()
            if k.startswith("instruction_turn") and k.endswith("_natural")
        )

        all_done = True
        for t in turn_indices:
            out_path = output_root / f"{session_id}_LayerwiseMemory_turn{t}.png"
            if not out_path.exists():
                all_done = False
                break

        if all_done:
            print(f"[SKIP] All turns already done for {session_id}")
            continue

        orig_img_path = original_dir / f"{img_id}_turn0.png"
        if not orig_img_path.exists():
            print(f"[SKIP] Original image not found: {orig_img_path}")
            continue

        print(
            f"\n[LayerwiseMemory][{processed_idx}/{len(scenario_files)}] "
            f"session_id={session_id}"
        )
        print(f"  - scenario: {scn_path}")
        print(f"  - image   : {orig_img_path}")
        print(f"  - category: {category}")

        # ----------------------
        # Turn 0: initialize latent memory with original caption
        # ----------------------
        initial_prompt = (
            scn.get("original_caption")
            or scenario.get("caption_turn1_natural_location")
            or scenario.get("caption_turn1_bbox")
            or "A photo."
        )

        init_image = Image.open(orig_img_path).convert("RGB").resize(
            (args.resolution, args.resolution)
        )
        full_mask = create_mask(
            args.resolution, (0, 0, args.resolution, args.resolution)
        )

        args.counter += 1
        cur_image = run_pipeline(
            pipe_inpaint,
            args,
            prompt=initial_prompt,
            image=init_image,
            mask_image=full_mask,
            inpaint=True,
            cattn_masking=False,
            multi_query_disentanglement=False,
            alpha=args.alpha,
            object_gen=args.object_gen,
            utilize_cache=args.utilize_cache,
            sigma=args.sigma,
            remove_vanilla_ratio=args.remove_vanilla_ratio,
            new_generation=True,
            remove_checkbox=False,
            strength=TURN0_STRENGTH,
            add_suffix=False,
        )

        turn_indices = sorted(
            int(k.replace("instruction_turn", "").replace("_natural", ""))
            for k in scenario.keys()
            if k.startswith("instruction_turn") and k.endswith("_natural")
        )

        for t in turn_indices:
            inst_key = f"instruction_turn{t}_natural"
            prompt = scenario.get(inst_key, None)

            if prompt is None:
                print(f"  [WARN] turn{t}: '{inst_key}' not found → skip")
                continue

            bboxes_norm_raw: List[List[float]] = []

            if category == "E" and t == 1:
                for k in [f"occluder_turn{t}_bbox", f"occluder*_turn{t}_bbox"]:
                    bbox_norm = scn.get(k, None)
                    if bbox_norm is None:
                        continue
                    if not (isinstance(bbox_norm, list) and len(bbox_norm) == 4):
                        continue
                    bboxes_norm_raw.append(bbox_norm)

            elif category == "E" and t == 2:
                for k in [f"occluder_turn{t}_bbox", f"occluder*_turn{t}_bbox"]:
                    bbox_norm = scn.get(k, None)
                    if bbox_norm is None:
                        continue
                    if not (isinstance(bbox_norm, list) and len(bbox_norm) == 4):
                        continue
                    bboxes_norm_raw.append(bbox_norm)
                    break

            else:
                bbox_key_main = f"occluder_turn{t}_bbox"
                bbox_norm = scn.get(bbox_key_main, None)
                if (
                    bbox_norm is not None
                    and isinstance(bbox_norm, list)
                    and len(bbox_norm) == 4
                ):
                    bboxes_norm_raw.append(bbox_norm)

            use_black_mask = False
            bboxes_norm_valid: List[List[float]] = []
            has_nonzero = False

            for bn in bboxes_norm_raw:
                if not (isinstance(bn, list) and len(bn) == 4):
                    continue
                bboxes_norm_valid.append(bn)
                if any(float(c) != 0.0 for c in bn):
                    has_nonzero = True

            if not bboxes_norm_valid or not has_nonzero:
                use_black_mask = True

            if use_black_mask:
                mask_img = Image.new("L", (args.resolution, args.resolution), 0)
            else:
                bboxes_pixel: List[Tuple[int, int, int, int]] = []
                for bn in bboxes_norm_valid:
                    x1 = int(bn[0] * args.resolution)
                    y1 = int(bn[1] * args.resolution)
                    x2 = int(bn[2] * args.resolution)
                    y2 = int(bn[3] * args.resolution)
                    bboxes_pixel.append((x1, y1, x2, y2))

                mask_img = (
                    create_mask_from_bboxes(args.resolution, bboxes_pixel)
                    if len(bboxes_pixel) > 1
                    else create_mask(args.resolution, bboxes_pixel[0])
                )

            if category in ["D", "E"] and t == 2:
                is_remove = True
            else:
                is_remove = False

            args.counter += 2

            out_path = output_root / f"{session_id}_LayerwiseMemory_turn{t}.png"

            if out_path.exists():
                try:
                    cur_image = Image.open(out_path).convert("RGB")
                    print(f"  [SKIP] turn{t} already exists → {out_path}")
                    continue
                except OSError:
                    print(
                        f"  [WARN] turn{t}: cached image is corrupted, re-generating → {out_path}"
                    )
                    try:
                        out_path.unlink()
                    except OSError:
                        pass

            edited = run_pipeline(
                pipe_inpaint,
                args,
                prompt=prompt,
                image=cur_image,
                mask_image=mask_img,
                inpaint=True,
                cattn_masking=args.cattn_masking,
                multi_query_disentanglement=args.multi_query_disentanglement,
                alpha=args.alpha,
                object_gen=args.object_gen,
                utilize_cache=args.utilize_cache,
                subject_token_idx=None,
                sigma=args.sigma,
                remove_vanilla_ratio=args.remove_vanilla_ratio,
                new_generation=True,
                remove_checkbox=is_remove,
                strength=EDIT_STRENGTH,
                add_suffix=(t == 1),
            )

            edited.save(out_path)
            cur_image = edited

            print(f"  [OK] turn{t} saved → {out_path}")

        del cur_image
        args.latent_memory.clear()

        if processed_idx % CACHE_CLEAR_INTERVAL == 0:
            clean_memory()
            print(
                f"  [MEMORY] Cleaned GPU cache at scenario {processed_idx}"
            )

    print("[LayerwiseMemory] All scenarios processed. Final memory cleanup...")
    clean_memory()


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "NY-BENCH: Multi-turn editing baseline "
            "LayerwiseMemory (Kim et al. [10])."
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=str(ROOT_DIR / "data"),
        help="Root directory containing images/ and scenarios/ (default: ./data)",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=str,
        default="scenarios",
        help="Folder name for scenario JSON files (default: scenarios)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Folder name for input images (default: images)",
    )
    parser.add_argument(
        "--pred-root",
        type=str,
        default=str(ROOT_DIR / "predictions"),
        help="Output directory for inference results (default: ./predictions)",
    )
    parser.add_argument(
        "--gpu-idx",
        type=int,
        default=0,
        help="GPU index to use (default: 0)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards (default: 1)",
    )
    parser.add_argument(
        "--shard-idx",
        type=int,
        default=0,
        help="Index of this shard (default: 0)",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    data_root = Path(args.data_root)
    scenario_root = data_root / args.scenarios_dir
    original_img_root = data_root / args.images_dir
    pred_root = Path(args.pred_root)

    if not scenario_root.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_root}")
    if not original_img_root.exists():
        raise FileNotFoundError(f"Images directory not found: {original_img_root}")

    run_layerwisememory_pipeline(
        scenario_root=scenario_root,
        original_img_root=original_img_root,
        base_results_root=pred_root,
        gpu_idx=args.gpu_idx,
        num_shards=args.num_shards,
        shard_idx=args.shard_idx,
    )


if __name__ == "__main__":
    main()
