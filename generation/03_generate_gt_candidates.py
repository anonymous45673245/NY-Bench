#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generation/03_generate_gt_candidates.py

[Role]
Generates 'Candidate' Ground Truth images using **Parallel Construction**.
This script creates the raw materials for the benchmark.

[Method: Parallel Construction]
Instead of sequential editing (Original -> Turn 1 -> Turn 2), we treat every state as a fresh "Add" task on the Original Image.

1. Turn 1 GT:
   - Draw RED box (occluder_turn1_bbox) on Original Image.
   - Prompt: "Add [occluder_turn1] to fill the red bounding box."
   - (Category E): "Add [obj1] to fill the red bounding box, and add [obj2] to fill the blue bounding box."

2. Turn 2 GT (Parallel):
   - Category A, B, C (Shrink, Move, Replace):
     - Draw GREEN box (occluder_turn2_bbox) on **Original Image**.
     - Prompt: "Add [occluder_turn2] to fill the green bounding box."
   
   - Category D (Remove):
     - Result is simply the Original Image.
   
   - Category E (Add Two -> Remove One):
     - Draw BLUE box (ccluder*_turn1_bbox) on **Original Image**.
     - Prompt: "Add [occluder*_turn1] to fill the blue bounding box."

[Note]
These generated images are CANDIDATES. Rigorous human verification/refinement is required for the final benchmark.
"""

import os
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
from io import BytesIO
from dotenv import load_dotenv
import io

import google.generativeai as genai
from google.api_core import retry

# --- [1] Utility Functions ---
def denormalize_bbox(bbox_norm, w, h):
    """Convert 0-1 normalized coordinates to pixel coordinates."""
    if not bbox_norm or bbox_norm == [0, 0, 0, 0]:
        return None
    return [
        bbox_norm[0] * w,
        bbox_norm[1] * h,
        bbox_norm[2] * w,
        bbox_norm[3] * h
    ]

def draw_bbox(image, bbox, color="red", width=5):
    """Draw BBox on the image."""
    if not bbox or bbox == [0, 0, 0, 0]:
        return image.copy()
    
    out_img = image.copy()
    draw = ImageDraw.Draw(out_img)
    draw.rectangle(bbox, outline=color, width=width)
    return out_img

class GeminiImageEditor:
    def __init__(self, api_key, model_name="gemini-2.5-flash-image"):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    @retry.Retry(predicate=retry.if_exception_type(Exception))
    def generate(self, marked_image: Image.Image, prompt: str) -> Image.Image:
        """
        Sends image and prompt, extracting the generated image from the response.
        """
        try:
            response = self.model.generate_content(
                [prompt, marked_image]
            )
            
            if not response.candidates or not response.parts:
                if response.prompt_feedback:
                    print(f" > [Block] Feedback: {response.prompt_feedback}")
                return None

            # [Modified parsing logic]
            for part in response.parts:
                
                # 1. If image data exists (check inline_data)
                if hasattr(part, 'inline_data') and part.inline_data:
                    if part.inline_data.mime_type.startswith("image/"):
                        image_data = part.inline_data.data
                        return Image.open(io.BytesIO(image_data))
                
                # 2. If text exists (Model refused or explained)
                if hasattr(part, 'text') and part.text:
                    # Print a snippet of the text for debugging
                    print(f" > [Warn] Model returned TEXT instead of image: {part.text[:100]}...")

            return None
                
        except Exception as e:
            print(f" > [Error] API Call Failed: {e}")
            return None

# --- [3] Main Execution (Single Thread) ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios_dir", type=str, default="../data/scenarios")
    parser.add_argument("--images_dir", type=str, default="../data/images")
    parser.add_argument("--output_base_dir", type=str, default="../data/candidates")
    # Model name (Note: Ensure this model supports image generation)
    parser.add_argument("--model_name", type=str, default="models/gemini-2.5-flash-image") 
    parser.add_argument("--api_key", type=str, help="Gemini API Key")
    args = parser.parse_args()

    load_dotenv()

    # Setup API Key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is required.")
        return
    
    # Initialize Editor
    try:
        editor = GeminiImageEditor(api_key, args.model_name)
        print(f"Initialized Model: {args.model_name}")
    except Exception as e:
        print(f"Model Initialization Error: {e}")
        return

    # Load scenario file list
    scenario_files = sorted(list(Path(args.scenarios_dir).glob("*.json")))
    print(f"Found {len(scenario_files)} scenarios. Starting Generation (Single Thread)...")

    # Process with tqdm
    for json_file in tqdm(scenario_files, desc="Processing"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            session_id = data.get('session_id', json_file.stem)
            image_id = session_id.split('_')[0]
            
            # --- Find Image Path ---
            img_path = Path(data.get('original_image', ''))
            if not img_path.exists():
                p1 = Path(args.images_dir) / session_id / "turn0.png"
                p2 = Path(args.images_dir) / image_id / "turn0.png"
                if p1.exists(): img_path = p1
                elif p2.exists(): img_path = p2
                else:
                    candidates = sorted(list(Path(args.images_dir).glob(f"{image_id}_*/turn0.png")))
                    if candidates: img_path = candidates[0]

            if not img_path.exists():
                # print(f"Image not found for {session_id}")
                continue

            # Load Original Image
            I0 = Image.open(img_path).convert("RGB")
            W, H = I0.size

            # Output Directory
            out_dir = Path(args.output_base_dir) / session_id
            out_dir.mkdir(parents=True, exist_ok=True)

            # --- Extract Data & Generate ---
            cat = data.get('category')
            
            # Coordinate conversion (0~1 -> Pixel)
            bbox_t1 = denormalize_bbox(data.get('occluder_turn1_bbox'), W, H)
            bbox_t2 = denormalize_bbox(data.get('occluder_turn2_bbox'), W, H)
            bbox_sec = denormalize_bbox(data.get('occluder*_turn1_bbox'), W, H)
            
            obj_t1 = data.get('occluder_turn1')
            obj_t2 = data.get('occluder_turn2')
            obj_sec = data.get('occluder*_turn1')

            if not cat: continue

            # ---------------------------------------------------------
            # 1. Turn 1 Generation
            # ---------------------------------------------------------
            t1_out_path = out_dir / "candidate_turn1.png"
            if not t1_out_path.exists():
                # Draw Red Box
                marked_I0_t1 = draw_bbox(I0, bbox_t1, color="red")
                prompt_t1 = f"Add {obj_t1} to fill the red bounding box."
                
                if cat == 'E' and bbox_sec:
                    marked_I0_t1 = draw_bbox(marked_I0_t1, bbox_sec, color="blue")
                    prompt_t1 += f", and add {obj_sec} to fill the blue bounding box."

                cand_t1 = editor.generate(marked_I0_t1, prompt_t1)
                if cand_t1:
                    cand_t1.save(t1_out_path)
                
                time.sleep(1) # Rate Limit

            # ---------------------------------------------------------
            # 2. Turn 2 Generation
            # ---------------------------------------------------------
            t2_out_path = out_dir / "candidate_turn2.png"
            if not t2_out_path.exists():
                cand_t2 = None
                
                if cat == 'D': # Remove -> Keep original
                    cand_t2 = I0.copy()
                
                elif cat == 'E': # Add Two -> Remove One (Only Blue Box remains)
                    prompt_t2 = f"Add {obj_sec} to fill the blue bounding box."
                    marked_I0_t2 = draw_bbox(I0, bbox_sec, color="blue") 
                    cand_t2 = editor.generate(marked_I0_t2, prompt_t2)
                
                elif cat in ['A', 'B', 'C']: # Shrink, Move, Replace (Green Box)
                    prompt_t2 = f"Add {obj_t2} to fill the green bounding box."
                    marked_I0_t2 = draw_bbox(I0, bbox_t2, color="green")
                    cand_t2 = editor.generate(marked_I0_t2, prompt_t2)

                if cand_t2:
                    cand_t2.save(t2_out_path)
                
                time.sleep(1) # Rate Limit

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")

    print("\n[Done] All scenarios processed.")

if __name__ == "__main__":
    main()