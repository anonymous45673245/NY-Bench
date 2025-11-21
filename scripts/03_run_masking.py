#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/03_run_masking.py

Batch runner for SAM segmentation.
Updated to work with the functional-style nybench.detection.sam_wrapper.
"""

import sys
import argparse
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from nybench.paths import NYBenchPaths
    # Import helpers from your new wrapper
    from nybench.detection.sam_wrapper import (
        resolve_sam_path,
        ensure_import_segment_anything,
        parse_box
    )
except ImportError as e:
    print(f"[FATAL] Could not import nybench modules. Error: {e}")
    sys.exit(1)


# -------------------------
# UTILITIES
# -------------------------

def make_black_mask(image_path: Path, mask_path: Path) -> None:
    """Create a black mask (all zeros) with the same size as the input image."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(image_path) as img:
            w, h = img.size
        black = Image.new("L", (w, h), 0)
        black.save(mask_path)
    except Exception as e:
        print(f"[ERROR] Failed to create black mask for {image_path.name}: {e}")


def run_sam_inference(
    predictor,
    image_path: Path,
    bbox_norm: List[float],
    target_mask_path: Path
) -> bool:
    """
    Run inference using a pre-loaded SAM predictor.
    bbox_norm: [x1, y1, x2, y2] normalized (0~1)
    """
    target_mask_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Image (OpenCV)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return False
    
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        # 2. Convert Box (Normalized -> Pixel)
        box_px = parse_box(tuple(bbox_norm), as_pixels=False, w=w, h=h)
        
        # SAM expects box shape (1, 4)
        box_tensor = box_px[None, :]

        # 3. Set Image & Predict
        predictor.set_image(image_rgb)
        
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_tensor,
            multimask_output=True 
        )
        
        # Select best mask based on score
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx].astype(np.uint8) # 0 or 1

        # 4. Save (0/1 -> 0/255)
        mask_255 = best_mask * 255
        cv2.imwrite(str(target_mask_path), mask_255)
        return True

    except Exception as e:
        print(f"[ERROR] SAM inference failed for {image_path.name}: {e}")
        return False


def make_mask_filename(session_id: str, model: str, turn_idx: int, occluder_id: int) -> str:
    base = f"{session_id}_{model}_turn{turn_idx}"
    if occluder_id == 0:
        return f"{base}_mask.png"
    else:
        return f"{base}_o1_mask.png"


# -------------------------
# MAIN PROCESSING
# -------------------------

def process_model(
    model_name: str,
    paths: NYBenchPaths,
    predictor: Any, # SamPredictor instance
    force: bool
):
    print(f"\n=== Processing Model: {model_name} ===")
    
    json_dir = paths.get_detection_dir(model_name)
    mask_dir = paths.get_mask_dir(model_name)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if not json_dir.exists():
        print(f"[WARN] Detection JSON directory not found: {json_dir}")
        return

    json_files = sorted(list(json_dir.glob("*.json")))
    if not json_files:
        print(f"[WARN] No JSON files found in {json_dir}")
        return

    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session_id = data.get("session_id")
        changed = False
        
        # Scene 5 check
        is_s5 = session_id.split("_")[-1] == "s5"
        target_ids = {0}
        if is_s5:
            target_ids.add(1)

        for turn in data.get("turns", []):
            turn_idx = turn.get("turn_index")
            
            # Resolve Image Path
            if turn_idx == 0:
                image_path = paths.images_root / session_id / "turn0.png"
            else:
                image_path = paths.get_prediction_image_path(model_name, session_id, turn_idx)
            
            if not image_path.exists():
                if Path(turn.get("image_path", "")).exists():
                     image_path = Path(turn["image_path"])
                else:
                    if turn_idx == 0: 
                        print(f"[SKIP] Turn 0 image missing: {image_path}")
                    continue

            # Process Occluders
            for oc in turn.get("occluders", []):
                oc_id = oc.get("occluder_id")
                if oc_id not in target_ids:
                    continue

                # Check existing
                existing_path = oc.get("mask_path")
                if not force and existing_path and Path(existing_path).exists():
                    continue

                mask_filename = make_mask_filename(session_id, model_name, turn_idx, oc_id)
                final_mask_path = mask_dir / mask_filename
                
                # --- Logic ---
                bbox = oc.get("bounding_box")
                oc_text = oc.get("occluder_text", "")
                is_none = isinstance(oc_text, str) and oc_text.lower() == "none"
                
                should_be_black = (turn_idx == 0) or is_none or (not bbox) or (sum(bbox) == 0)

                if should_be_black:
                    make_black_mask(image_path, final_mask_path)
                    oc["mask_path"] = str(final_mask_path.relative_to(PROJECT_ROOT))
                    changed = True
                    continue

                # Run SAM
                success = run_sam_inference(predictor, image_path, bbox, final_mask_path)
                
                if success:
                    oc["mask_path"] = str(final_mask_path.relative_to(PROJECT_ROOT))
                    changed = True
                    print(f" -> Generated: {mask_filename}")
                else:
                    make_black_mask(image_path, final_mask_path)
                    oc["mask_path"] = str(final_mask_path.relative_to(PROJECT_ROOT))
                    changed = True

        if changed:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run SAM Masking for NY-BENCH")
    parser.add_argument("--models", nargs="+", required=True, help="Models to process")
    parser.add_argument("--sam-ckpt", type=Path, required=True, help="Path to SAM checkpoint")
    parser.add_argument("--sam-dir", type=str, default=None, help="Path to SAM source code")
    parser.add_argument("--model-type", type=str, default="vit_h", help="SAM model type")
    parser.add_argument("--force", action="store_true", help="Overwrite existing masks")
    args = parser.parse_args()

    # 1. Initialize Paths (FIXED: Added data_root argument)
    paths = NYBenchPaths(data_root=PROJECT_ROOT / "data")

    # 2. Setup SAM Environment
    print("Setting up SAM environment...")
    sam_dir_resolved = resolve_sam_path(args.sam_dir)
    ensure_import_segment_anything(sam_dir_resolved)

    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print("[FATAL] Failed to import segment_anything.")
        sys.exit(1)

    # 3. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ({args.model_type}) from {args.sam_ckpt} to {device}...")
    
    if not args.sam_ckpt.exists():
        print(f"[FATAL] Checkpoint not found: {args.sam_ckpt}")
        sys.exit(1)

    try:
        sam = sam_model_registry[args.model_type](checkpoint=str(args.sam_ckpt))
        sam.to(device=device)
        predictor = SamPredictor(sam)
        print("SAM loaded successfully.")
    except Exception as e:
        print(f"[FATAL] Error loading SAM model: {e}")
        sys.exit(1)

    # 4. Run Batch
    for model in args.models:
        process_model(model, paths, predictor, args.force)
    
    print("\n--- Masking Complete ---")

if __name__ == "__main__":
    main()