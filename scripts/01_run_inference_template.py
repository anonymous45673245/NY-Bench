#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/01_run_inference_template.py

Standard inference template for NY-BENCH.
This script demonstrates how to:
1. Load the dataset using NYBenchDataset.
2. Iterate through multi-turn editing scenarios.
3. Apply bounding boxes (optional visual prompting).
4. Save results using the standardized NYBenchPaths.

[Usage]
    python scripts/01_run_inference_template.py --model_name MyModel

[Customization]
    Users should implement the `edit_image` function to connect their own model/API.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw
from tqdm import tqdm

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from nybench.dataset import NYBenchDataset
from nybench.paths import NYBenchPaths


# ---------------------------------------------------------
# 1. Helper Utilities (Optional)
# ---------------------------------------------------------

def draw_bbox(image: Image.Image, bbox: List[float], color: str = "red", width: int = 5) -> Image.Image:
    """
    Draws a bounding box on the image. Used for visual prompting models (e.g., GPT-4V, Gemini).
    bbox format: [x1, y1, x2, y2] (absolute pixel coordinates)
    """
    if not bbox or sum(bbox) == 0:
        return image.copy()
    
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle(bbox, outline=color, width=width)
    return img_draw


# ---------------------------------------------------------
# 2. Model Inference Logic (TODO: Implement this)
# ---------------------------------------------------------

def edit_image(
    input_image: Image.Image, 
    instruction: str, 
    bbox: Optional[List[float]] = None,
    turn_idx: int = 1
) -> Image.Image:
    """
    [USER IMPLEMENTATION REQUIRED]
    Run your model inference here.

    Args:
        input_image: The PIL Image to edit.
        instruction: The text instruction.
        bbox: The target bounding box [x1, y1, x2, y2] (pixels). None if not provided.
        turn_idx: Current turn index (1 or 2). Useful for changing box colors.

    Returns:
        PIL.Image: The edited result image.
    """
    
    # --- Example Logic for Visual Prompting Models ---
    # 1. Draw BBox if needed (Red for T1, Green for T2 is a common convention)
    color = "red" if turn_idx == 1 else "green"
    visual_prompt_image = draw_bbox(input_image, bbox, color=color) if bbox else input_image
    
    # 2. Prepare Instruction (Optional: Add safety prompts)
    full_instruction = instruction
    # full_instruction += " Do not change the background."

    # 3. Call your Model / API
    # response = my_model.generate(image=visual_prompt_image, text=full_instruction)
    # return response.image
    
    # --- Placeholder: Return Identity (for testing) ---
    print(f" [Mock Inference] Turn {turn_idx} | Instr: {instruction[:30]}... | BBox: {bbox}")
    return input_image 


# ---------------------------------------------------------
# 3. Main Pipeline
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NY-BENCH Inference Template")
    parser.add_argument("--model_name", type=str, required=True, help="Name of your model (e.g., GPT-4o)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
    args = parser.parse_args()

    # 1. Initialize Dataset & Paths
    # load_images=True ensures we get PIL objects directly
    dataset = NYBenchDataset(load_images=True)
    paths = NYBenchPaths()
    
    # 2. Setup Output Directory
    output_dir = paths.get_prediction_dir(args.model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output Directory: {output_dir}")

    # 3. Iterate Scenarios
    for i in tqdm(range(len(dataset)), desc=f"Running {args.model_name}"):
        sample = dataset[i]
        session_id = sample['session_id']
        
        # Load Scenario Metadata (Instructions, BBoxes)
        scenario_json_path = sample['paths']['scenario_json']
        with open(scenario_json_path, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
            
        scenario_info = scenario_data.get("scenario", {})
        
        # =========================================
        # TURN 1
        # =========================================
        save_path_t1 = paths.get_prediction_image_path(args.model_name, session_id, 1)
        
        # Skip if exists
        if save_path_t1.exists() and not args.overwrite:
            # Load existing result to use as input for Turn 2
            output_t1 = Image.open(save_path_t1).convert("RGB")
        else:
            # Input: Original Turn 0 Image
            input_t0 = sample['images']['turn0']
            instr_t1 = scenario_info.get('instruction_turn1', "")
            bbox_t1 = scenario_data.get('occluder_turn1_bbox', None)
            
            if not instr_t1: continue # Skip invalid data

            # Run Inference
            output_t1 = edit_image(input_t0, instr_t1, bbox=bbox_t1, turn_idx=1)
            
            # Save
            output_t1.save(save_path_t1)

        # =========================================
        # TURN 2
        # =========================================
        save_path_t2 = paths.get_prediction_image_path(args.model_name, session_id, 2)
        
        instr_t2 = scenario_info.get('instruction_turn2', "")
        if not instr_t2:
            continue # No Turn 2 for this scenario

        if save_path_t2.exists() and not args.overwrite:
            continue
        
        # Input: Output of Turn 1
        input_t1 = output_t1 
        bbox_t2 = scenario_data.get('occluder_turn2_bbox', None)

        # Run Inference
        output_t2 = edit_image(input_t1, instr_t2, bbox=bbox_t2, turn_idx=2)
        
        # Save
        output_t2.save(save_path_t2)

    print("\nInference complete!")


if __name__ == "__main__":
    main()