#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generation/01_generate_scenarios.py

[Role]
1. Scans the input directory for unique original images (turn0.png).
2. Utilizes a Multimodal LLM (Gemini) to analyze the image content.
3. Generates 5 distinct multi-turn editing scenarios (Categories A, B, C, D, E) per image.
   - Defines the 'Occludee' (target to be hidden) and 'Occluder' (object added).
   - Generates specific editing instructions for Turn 1 and Turn 2.

[Pipeline Note: Human Verification]
This script automates the *proposal* of editing scenarios.
In the actual benchmark construction process, these generated scenarios undergo 
rigorous HUMAN VERIFICATION and SELECTION.
Only scenarios that are logically consistent, physically plausible, and visually 
meaningful are selected for the final dataset.
"""

import os
import json
import time
import glob
import argparse
import google.generativeai as genai
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# --- User Configuration ---
SYSTEM_PROMPT = """
**Role & Objective:**
You are an expert in designing benchmark datasets for multi-turn image editing.
Your task is to analyze the provided **Original Image** and generate a JSON array containing **exactly 5 distinct scenarios** (one for each category: A, B, C, D, E).

---
### 1. Core Concepts & Object Selection
For each scenario, identify two key components based on the image content:

**A. The Occludee (Target Region)**
* **Definition:** The part of the original image that will be hidden in Turn 1 and revealed in Turn 2.
* **Requirement:** Must be a **salient, meaningful detail** (e.g., "owl's chest", "brand logo", "face of the statue").
* **Restriction:** Do NOT choose empty space, plain background, or insignificant details.

**B. The Occluder (Added Object)**
* **Definition:** The new object added in Turn 1 to hide the Occludee.
* **Requirement:** Must be "Bulky" enough to effectively cover the target.
* **Examples:** (Good) "a teddy bear", "a thick book", "a bouquet". (Bad) "a pencil", "a wire".

---
### 2. Scenario Categories (Generate 1 of each)

* **Category A (Add -> Shrink):**
    * T1: Add an object to cover the occludee.
    * T2: Shrink that object (revealing the background).
* **Category B (Add -> Move):**
    * T1: Add an object to cover the occludee.
    * T2: Move that object to a different location.
* **Category C (Add -> Replace):**
    * T1: Add an object to cover the occludee.
    * T2: Replace it with a **smaller** object (revealing part of the background).
* **Category D (Add -> Remove):**
    * T1: Add an object to cover the occludee.
    * T2: Remove that object completely.
* **Category E (Add Two -> Remove One):**
    * T1: Add **two** objects. Object 1 covers the occludee; Object 2 is elsewhere.
    * T2: Remove **only** Object 1 (the one covering the occludee).

---
### 3. JSON Field Logic (Strict Rules)
You must populate the object name fields based on the category rules below:

**1) `occluder_turn1` (Primary Object)**
* Always the name of the main object added in Turn 1.

**2) `occluder_turn2` (State in Turn 2)**
* **Category A (Shrink) & B (Move):** Must be the **SAME** string as `occluder_turn1`.
* **Category C (Replace):** Must be the **NEW** object name (the smaller one).
* **Category D (Remove) & E (Remove):** Must be `null` (since the object is gone).

**3) `occluder*_turn1` & `occluder*_turn2`**
* **Category E (Add Two):** * `occluder*_turn1`: Name of the second object (Blue box).
    * `occluder*_turn2`: Same as `occluder*_turn1` (it remains).
* **Category A, B, C, D:** Must be `null`.

---
### 4. Strict Instruction Templates
You **MUST** use the following templates for the instruction text. Do not alter the phrasing.

**[Turn 1 Instructions]**
* **For Categories A, B, C, D:**
    "Add [object] to fill the red bounding box."
* **For Category E (Multi-Add):**
    "Add [object 1] to fill the red bounding box and add [object 2] to fill the blue bounding box"

**[Turn 2 Instructions]**
* **For Categories A, B, C (Modify/Move):**
    "[Action] the [object] to fit in the green bounding box."
    *(Action examples: Shrink, Move, Replace)*
* **For Categories D, E (Remove):**
    "Remove the [object]."

---
### 5. Output Format
Return **ONLY** a single valid JSON array containing 5 objects. No markdown formatting or extra text.

**JSON Structure:**
[
  {
    "category": "A",
    "occludee": "description of the hidden region",
    "occluder_turn1": "object name",
    "occluder_turn2": "object name",
    "occluder*_turn1": null,
    "occluder*_turn2": null,
    "scenario": {
      "instruction_turn1": "Add a [obj] to fill the red bounding box.",
      "instruction_turn2": "Shrink the [obj] to fit in the green bounding box."
    }
  },
  ... (Total 5 items: A, B, C, D, E)
]
"""

def generate_scenarios_for_image(model, image_path):
    """
    Calls Gemini API with the image and system prompt.
    """
    img = Image.open(image_path)
    
    try:
        response = model.generate_content([SYSTEM_PROMPT, img])
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(text_response)
    except Exception as e:
        print(f"[Error] Failed to generate for {image_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../data/images", help="Directory containing source images")
    parser.add_argument("--output_dir", type=str, default="../data/scenarios", help="Directory to save JSONs")
    parser.add_argument("--api_key", type=str, help="Gemini API Key")
    args = parser.parse_args()

    load_dotenv()

    # Setup Gemini
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: API Key required.")
        return
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-image')

    os.makedirs(args.output_dir, exist_ok=True)

    all_image_files = sorted(list(Path(args.input_dir).rglob("turn0.png")))

    unique_images = {}
    for img_path in all_image_files:
        folder_name = img_path.parent.name
        image_id = folder_name.split('_')[0] # "0001"
        
        if image_id not in unique_images:
            unique_images[image_id] = img_path

    target_files = list(unique_images.values())

    print(f"Processing {len(target_files)} unique images out of {len(all_image_files)} folders.")

    for img_path in tqdm(target_files):
        folder_name = img_path.parent.name
        image_id = folder_name.split('_')[0] # "0001"

        if (Path(args.output_dir) / f"{image_id}_s1.json").exists():
            continue

        scenarios = generate_scenarios_for_image(model, img_path)
        
        if scenarios:
            for idx, scenario_data in enumerate(scenarios, 1):
                
                new_session_id = f"{image_id}_s{idx}"
                output_path = Path(args.output_dir) / f"{new_session_id}.json"

                with open(output_path, 'w') as f:
                    json.dump({
                        "session_id": new_session_id,
                        "image_id": image_id,         
                        "original_image": str(img_path),
                        **scenario_data 
                    }, f, indent=2)
            
            # Rate Limit Handling
            time.sleep(2) 

if __name__ == "__main__":
    main()