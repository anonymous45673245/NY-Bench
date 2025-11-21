#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_run_inference_gpt.py

Multi-turn image editing inference pipeline for NY-BENCH using GPT (OpenAI).

Inputs:
    - data/images/<session_id>/turn0.png
    - data/scenarios/<session_id>.json

Outputs:
    - predictions/GPT/<session_id>_GPT_turn1.png
    - predictions/GPT/<session_id>_GPT_turn2.png  (if turn2 exists)

Example:
    python scripts/01_run_inference_gpt.py \
        --model-id gpt-5.1 \
        --max-workers 8
"""

import os
import sys
import json
import base64
import time
import argparse
import concurrent.futures
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# ---------------------------------------------------------
# Path setup relative to the project root
# ---------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.append(str(ROOT_DIR))

# Utility functions assumed to exist in nybench/utils/
from nybench.utils.image_utils import draw_bbox_on_image
from nybench.utils.io_utils import encode_pil_image_to_base64


# ---------------------------------------------------------
# 1. Internal OpenAI helper functions
# ---------------------------------------------------------

def _save_image_from_response(response, output_path: str) -> bool:
    """
    Extracts generated image (base64) from OpenAI Responses API output
    and saves it to `output_path`. Returns True on success.

    NOTE:
        The exact structure of `response.output` may vary depending on
        the OpenAI Python SDK version. Adjust this function accordingly.
    """
    try:
        image_b64 = None

        # Example extraction logic (modify as needed for your SDK version)
        for out in getattr(response, "output", []):
            if getattr(out, "type", "") == "image_generation":
                image_b64 = out.image_base64
                break

        if not image_b64:
            print("               > Warning: No image data found in response.")
            return False

        img_bytes = base64.b64decode(image_b64)
        with open(output_path, "wb") as f:
            f.write(img_bytes)

        print(f"               > Saved: {output_path}")
        return True

    except Exception as e:
        print(f"               > Error parsing image from response: {e}")
        return False


def _process_single_scenario(
    filename: str,
    original_img_root: Path,
    scenario_root: Path,
    results_dir: Path,
    client: OpenAI,
    model_id: str,
):
    """
    Process a single scenario file (e.g., '0001_s0.json').
    Executed inside a thread worker.
    """
    session_id = Path(filename).stem
    log_prefix = f"[{session_id}]"

    print(f"\n--- {log_prefix} Start: {filename} ---")

    scenario_path = scenario_root / filename
    original_img_path = original_img_root / session_id / "turn0.png"

    turn1_output_path = results_dir / f"{session_id}_GPT_turn1.png"
    turn2_output_path = results_dir / f"{session_id}_GPT_turn2.png"

    if not original_img_path.exists():
        print(f" > Warning: Original image not found: {original_img_path}")
        return

    try:
        # Load scenario JSON
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario_data = json.load(f)

        scenario_info = scenario_data.get("scenario", {})
        instruction_turn1 = scenario_info.get("instruction_turn1")
        instruction_turn2 = scenario_info.get("instruction_turn2")

        has_turn2 = bool(instruction_turn2)

        # Skip if already processed
        if turn1_output_path.exists() and (not has_turn2 or turn2_output_path.exists()):
            print(f" > {log_prefix} Already processed. Skipping.")
            return

        if not instruction_turn1:
            print(f" > Warning: Missing instruction_turn1. Skipping.")
            return

        # Append safety rules for bounding boxes
        instruction_turn1 += (
            " Do not replace or delete any existing objects in the image."
        )
        instruction_turn1 += (
            " The red bounding box is only for indicating the editing area; "
            "do not include the red bounding box in the final output image."
        )

        if instruction_turn2 and "green bounding box" in instruction_turn2.lower():
            instruction_turn2 += (
                " The green bounding box is only for indicating the editing area; "
                "do not include it in the final output image."
            )
            print(f"          > {log_prefix} Added bbox-removal rule for turn2.")

        # ------------------------------------------
        # Turn 1: prepare image with red bbox
        # ------------------------------------------
        original_image_pil = Image.open(original_img_path).convert("RGB")
        occluder_turn1_bbox = scenario_data.get("occluder_turn1_bbox")

        if (
            occluder_turn1_bbox
            and isinstance(occluder_turn1_bbox, list)
            and len(occluder_turn1_bbox) == 4
        ):
            image_for_turn1_pil = draw_bbox_on_image(
                original_image_pil, occluder_turn1_bbox, color="red"
            )
        else:
            image_for_turn1_pil = original_image_pil

        base64_image_for_turn1 = encode_pil_image_to_base64(
            image_for_turn1_pil, format="PNG"
        )

        print(f"          > {log_prefix} Sending turn1 request...")
        response1 = client.responses.create(
            model=model_id,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction_turn1},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image_for_turn1}",
                        },
                    ],
                }
            ],
            tools=[{"type": "image_generation", "size": "1024x1024"}],
        )

        if not _save_image_from_response(response1, str(turn1_output_path)):
            print(f"     > {log_prefix} Turn1 failed. Aborting scenario.")
            return

        # ------------------------------------------
        # Turn 2 (if exists)
        # ------------------------------------------
        if instruction_turn2:
            turn1_result_pil = Image.open(turn1_output_path).convert("RGB")
            occluder_turn2_bbox = scenario_data.get("occluder_turn2_bbox")

            if (
                occluder_turn2_bbox
                and isinstance(occluder_turn2_bbox, list)
                and len(occluder_turn2_bbox) == 4
            ):
                image_for_turn2_pil = draw_bbox_on_image(
                    turn1_result_pil, occluder_turn2_bbox, color="green"
                )
            else:
                image_for_turn2_pil = turn1_result_pil

            base64_image_for_turn2 = encode_pil_image_to_base64(
                image_for_turn2_pil, format="PNG"
            )

            print(f"          > {log_prefix} Sending turn2 request...")
            response2 = client.responses.create(
                model=model_id,
                previous_response_id=response1.id,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": instruction_turn2},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{base64_image_for_turn2}",
                            },
                        ],
                    }
                ],
                tools=[{"type": "image_generation", "size": "1024x1024"}],
            )

            _save_image_from_response(response2, str(turn2_output_path))

        print(f"     > {log_prefix} Done.")
        time.sleep(1)  # Simple rate-limit buffer

    except Exception as e:
        print(f" > {log_prefix} Unexpected error: {e}")
        return

    return filename


# ---------------------------------------------------------
# 2. Pipeline entry function
# ---------------------------------------------------------

def run_gpt_pipeline(
    scenario_root: Path,
    original_img_root: Path,
    base_results_root: Path,
    model_id: str = "gpt-5.1",
    max_workers: int = 8,
):
    """
    Full GPT inference pipeline for NY-BENCH.
    """
    load_dotenv()

    try:
        client = OpenAI()  # API key loaded from environment
    except Exception as e:
        print(f"OpenAI client initialization failed: {e}")
        return

    results_dir = base_results_root / "GPT"
    results_dir.mkdir(parents=True, exist_ok=True)

    scenario_files = sorted(
        f for f in os.listdir(scenario_root) if f.endswith(".json")
    )

    print(f"--- GPT Inference Pipeline ---")
    print(f"Total scenarios: {len(scenario_files)}")
    print(f"Threads: {max_workers}")
    print(f"Output directory: {results_dir}")
    print(f"Model: {model_id}")

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for filename in scenario_files:
            futures.append(
                executor.submit(
                    _process_single_scenario,
                    filename,
                    original_img_root,
                    scenario_root,
                    results_dir,
                    client,
                    model_id,
                )
            )

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                if future.result():
                    completed += 1
            except Exception as e:
                print(f"[Error] Worker failed: {e}")

    print(f"\n--- Pipeline Complete ---")
    print(f"Completed: {completed} / {len(futures)}")


# ---------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="NY-BENCH: Multi-turn editing with OpenAI GPT"
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
        "--model-id",
        type=str,
        default="gpt-5.1",
        help="OpenAI model ID (e.g. gpt-5.1)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel threads",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    scenario_root = data_root / args.scenarios_dir
    original_img_root = data_root / args.images_dir
    pred_root = Path(args.pred_root)

    if not scenario_root.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_root}")
    if not original_img_root.exists():
        raise FileNotFoundError(f"Images directory not found: {original_img_root}")

    run_gpt_pipeline(
        scenario_root=scenario_root,
        original_img_root=original_img_root,
        base_results_root=pred_root,
        model_id=args.model_id,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
