#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_run_inference_mcedit.py

Multi-turn image editing inference pipeline for NY-BENCH using MC-Edit
(the FluxEditor-based baseline from ICCV).

We assume the external MC-Edit repository is placed under:
    <project_root>/models/MCEdit/

Inputs:
    - data/scenarios/<session_id>.json
    - data/images_flat/<base_id>_turn0.png   (or any directory you point to)

Outputs:
    - predictions/MCEdit/<session_id>_MCEdit_turn1.png
    - predictions/MCEdit/<session_id>_MCEdit_turn2.png

Example:
    python scripts/01_run_inference_mcedit.py \
        --data-root ./data \
        --images-dir images_flat \
        --pred-root ./predictions
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image, ImageDraw

# -------------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

# External MC-Edit repo root (FluxEditor implementation)
MCEDIT_ROOT = ROOT_DIR / "models" / "MCEdit"
sys.path.insert(0, str(MCEDIT_ROOT))

try:
    from gradio_demo_playground import FluxEditor  # type: ignore
except ImportError:
    print("=" * 50)
    print("Error: Could not import 'FluxEditor' from 'gradio_demo_playground.py'.")
    print("Make sure the MC-Edit repository is located at:")
    print(f"    {MCEDIT_ROOT}")
    print("and that 'gradio_demo_playground.py' is inside that directory.")
    print("=" * 50)
    sys.exit(1)


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def draw_bbox_on_image(
    image: Image.Image,
    bbox: List[float],
    color: str = "red",
    width: int = 5,
) -> Image.Image:
    """
    Draw a bounding box on a PIL image.

    bbox is [xmin, ymin, xmax, ymax] in normalized coordinates (0~1).
    """
    image_with_box = image.copy()
    draw = ImageDraw.Draw(image_with_box)

    img_width, img_height = image.size
    xmin, ymin, xmax, ymax = bbox

    xmin = max(0.0, min(1.0, float(xmin)))
    ymin = max(0.0, min(1.0, float(ymin)))
    xmax = max(0.0, min(1.0, float(xmax)))
    ymax = max(0.0, min(1.0, float(ymax)))

    abs_xmin = int(xmin * img_width)
    abs_ymin = int(ymin * img_height)
    abs_xmax = int(xmax * img_width)
    abs_ymax = int(ymax * img_height)

    draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline=color, width=width)
    return image_with_box


# -------------------------------------------------------------------------
# Main MC-Edit pipeline
# -------------------------------------------------------------------------

def run_mcedit_pipeline(
    scenario_root: Path,
    original_img_root: Path,
    base_results_root: Path,
    device: Optional[str] = None,
    limit: Optional[int] = None,
):
    """
    Run the MC-Edit (FluxEditor-based) baseline on all scenarios in scenario_root.

    Args:
        scenario_root: Path to data/scenarios/
        original_img_root: Path to directory containing <base_id>_turn0.png
        base_results_root: Path to predictions/
        device: Device string for the model (e.g., 'cuda' or 'cpu'). If None, auto-detect.
        limit: If not None, process at most this many JSON files (for quick tests).
    """
    results_dir = base_results_root / "MCEdit"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(scenario_root.glob("*.json"))
    if limit is not None:
        json_files = json_files[:limit]

    if not json_files:
        print(f"Warning: No JSON files found in {scenario_root}")
        return

    # Build FluxEditor args (parsed without CLI)
    parser = argparse.ArgumentParser(description="MC-Edit (FluxEditor) Automation")
    parser.add_argument("--name", type=str, default="flux-dev")
    parser.add_argument(
        "--device",
        type=str,
        default=device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--offload", action="store_true", default=False)
    flux_args = parser.parse_args([])

    print("Loading FluxEditor model (MC-Edit)...")
    editor = FluxEditor(flux_args)
    print("✓ FluxEditor model loaded.")

    default_settings = {
        "editing_strategy": ["replace_v"],
        "denoise_strategy": "multi_turn_consistent",
        "num_steps": 15,
        "guidance": 3.5,
        "attn_guidance_start_block": 11,
        "inject_step": 1,
        "init_image_2": None,
    }

    print(f"Found {len(json_files)} scenario files. Starting inference loop...")

    for idx, json_file_path in enumerate(json_files, start=1):
        json_id = json_file_path.stem  # e.g., "0001_s0"
        base_id = json_id.split("_")[0]  # e.g., "0001"
        print(f"\n--- [{idx}/{len(json_files)}] Scenario: {json_id}.json ---")

        original_image_path = original_img_root / f"{base_id}_turn0.png"
        if not original_image_path.exists():
            print(f"    [Skip] Original image not found: {original_image_path}")
            continue

        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                scenario_obj = json.load(f)
        except json.JSONDecodeError:
            print(f"    [Error] Failed to parse JSON: {json_file_path}")
            continue

        editor.reset_history()
        print("    > Running scenario...")

        try:
            scenario_data = scenario_obj["scenario"]
            instruction1 = scenario_data["instruction_turn1"]
            instruction2 = scenario_data["instruction_turn2"]
            bbox1_coords = scenario_obj.get("occluder_turn1_bbox")
        except KeyError as e:
            print(f"    [Error] Missing key in scenario JSON: {e}. Skipping.")
            continue

        if not isinstance(bbox1_coords, list) or len(bbox1_coords) != 4:
            print("    [Warning] Invalid occluder_turn1_bbox. Skipping scenario.")
            continue

        turn0_image_pil = Image.open(original_image_path).convert("RGB")
        turn0_image_pil_with_box = draw_bbox_on_image(
            turn0_image_pil,
            bbox1_coords,
            color="red",
            width=3,
        )
        turn0_image_np = np.array(turn0_image_pil_with_box)

        print(f"     - Turn 1: target='{instruction1[:50]}...'")
        turn1_image_pil, _ = editor.process_image(
            init_image=turn0_image_np,
            source_prompt="",
            target_prompt=instruction1,
            **default_settings,
        )
        output_path_t1 = results_dir / f"{json_id}_MCEdit_turn1.png"
        turn1_image_pil.save(output_path_t1)
        print(f"     - Turn 1 saved: {output_path_t1}")

        bbox2_coords = scenario_obj.get("occluder_turn2_bbox")
        if not isinstance(bbox2_coords, list) or len(bbox2_coords) != 4:
            print("     - Warning: Invalid occluder_turn2_bbox. Skipping Turn 2.")
            continue

        print(f"     - Turn 2: source='{instruction1[:50]}...', target='{instruction2[:50]}...'")

        turn1_image_pil_with_box = draw_bbox_on_image(
            turn1_image_pil,
            bbox2_coords,
            color="green",
            width=3,
        )
        turn1_image_np = np.array(turn1_image_pil_with_box)

        turn2_image_pil, _ = editor.process_image(
            init_image=turn1_image_np,
            source_prompt="",
            target_prompt=instruction2,
            **default_settings,
        )
        output_path_t2 = results_dir / f"{json_id}_MCEdit_turn2.png"
        turn2_image_pil.save(output_path_t2)
        print(f"     - Turn 2 saved: {output_path_t2}")

    print("\nAll MC-Edit automation jobs are finished.")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="NY-BENCH: Multi-turn editing baseline MC-Edit (ICCV)."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=str(ROOT_DIR / "data"),
        help="Root directory containing scenarios/ and images/ (default: ./data)",
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
        default="images_flat",
        help=(
            "Folder containing <base_id>_turn0.png (default: images_flat). "
            "For example: 0001_turn0.png, 0002_turn0.png, ..."
        ),
    )
    parser.add_argument(
        "--pred-root",
        type=str,
        default=str(ROOT_DIR / "predictions"),
        help="Output directory for inference results (default: ./predictions)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for MC-Edit (e.g., 'cuda' or 'cpu'). Default: auto-detect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of scenarios to process (for quick tests).",
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

    run_mcedit_pipeline(
        scenario_root=scenario_root,
        original_img_root=original_img_root,
        base_results_root=pred_root,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
