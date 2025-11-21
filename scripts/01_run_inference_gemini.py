#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_run_inference_gemini.py

Multi-turn image editing inference pipeline for NY-BENCH using Gemini.

Inputs:
    - data/images/<session_id>/turn0.png
    - data/scenarios/<session_id>.json

Outputs:
    - predictions/Gemini/<session_id>_Gemini_turn1.png
    - predictions/Gemini/<session_id>_Gemini_turn2.png  (if turn2 exists)

Example:
    python scripts/01_run_inference_gemini.py \
        --model-id models/gemini-2.5-flash-image \
        --max-workers 8
"""

import os
import sys
import json
import time
import concurrent.futures
from pathlib import Path
from io import BytesIO

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.append(str(ROOT_DIR))

from nybench.utils.image_utils import draw_bbox_on_image


def _save_first_image_from_response(response, output_path: str) -> int:
    """
    Extracts the first valid image from a Gemini response and saves it.
    Returns 1 on success, 0 otherwise.
    """
    try:
        if not getattr(response, "candidates", None):
            print("               > Error: No candidates returned by the API.")
            if getattr(response, "prompt_feedback", None):
                print(f"               > Feedback: {response.prompt_feedback}")
            return 0

        candidate = response.candidates[0]
        if getattr(candidate, "finish_reason", None) != 1:
            print(f"               > Error: Generation stopped (finish_reason={candidate.finish_reason}).")
            return 0
    except Exception as e:
        print(f"               > Error while validating response: {e}")
        return 0

    for part in candidate.content.parts:
        if hasattr(part, "inline_data") and part.inline_data is not None and part.inline_data.data:
            try:
                image_bytes = part.inline_data.data
                image = Image.open(BytesIO(image_bytes))
                image.save(output_path)
                print(f"               > Saved: {output_path}")
                return 1
            except Exception as img_e:
                print(f"               > Warning: Failed to parse image data, trying next part... ({img_e})")

    print("               > Error: No valid image found in response.")
    return 0


def _process_single_scenario(
    filename: str,
    original_img_root: Path,
    scenario_root: Path,
    results_dir: Path,
    model: genai.GenerativeModel,
):
    """
    Processes a single scenario file (e.g., '0001_s0.json') in a worker thread.
    """
    session_id = Path(filename).stem
    log_prefix = f"[{session_id}]"

    print(f"\n--- {log_prefix} Start: {filename} ---")

    scenario_path = scenario_root / filename
    original_img_path = original_img_root / session_id / "turn0.png"

    if not original_img_path.exists():
        print(f" > Warning: Original image not found: {original_img_path}. Skipping.")
        return

    try:
        turn1_output_path = results_dir / f"{session_id}_Gemini_turn1.png"
        turn2_output_path = results_dir / f"{session_id}_Gemini_turn2.png"

        temp_instruction_turn2 = ""
        try:
            with open(scenario_path, "r", encoding="utf-8") as f_check:
                temp_data = json.load(f_check)
                temp_instruction_turn2 = temp_data.get("scenario", {}).get("instruction_turn2")
        except Exception:
            pass

        if turn1_output_path.exists() and (not temp_instruction_turn2 or turn2_output_path.exists()):
            print(f" > {log_prefix} Turn1/Turn2 results already exist. Skipping.")
            return

        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario_data = json.load(f)

        if not scenario_data:
            print(f" > {log_prefix} Warning: Empty JSON. Skipping.")
            return

        try:
            original_image_pil = Image.open(original_img_path).convert("RGB")
        except FileNotFoundError:
            print(f" > Warning: {log_prefix} Original image not found. Skipping.")
            return

        print(f" > {log_prefix} Validating scenario...")

        scenario = scenario_data.get("scenario", {})
        instruction_turn1 = scenario.get("instruction_turn1")
        instruction_turn2 = scenario.get("instruction_turn2")
        occluder_turn1_bbox = scenario_data.get("occluder_turn1_bbox")

        if not instruction_turn1:
            print(f"     > {log_prefix} Warning: Missing instruction_turn1. Skipping.")
            return

        if (
            not occluder_turn1_bbox
            or not isinstance(occluder_turn1_bbox, list)
            or len(occluder_turn1_bbox) != 4
        ):
            print(f"     > Warning: {log_prefix} Invalid occluder_turn1_bbox. Proceeding without bbox for turn1.")
            image_for_turn1 = original_image_pil
        else:
            image_for_turn1 = draw_bbox_on_image(
                original_image_pil, occluder_turn1_bbox, color="red", width=5
            )
            print(f"     > {log_prefix} Drew occluder_turn1_bbox {occluder_turn1_bbox} on the image.")

        print(f"     > {log_prefix} Starting Gemini calls...")

        try:
            chat = model.start_chat()

            print(f"     > {log_prefix} Sending turn1: {instruction_turn1[:50]}...")
            response1 = chat.send_message([instruction_turn1, image_for_turn1])

            if _save_first_image_from_response(response1, str(turn1_output_path)) == 0:
                print(f"     > {log_prefix} Turn1 image generation failed. Aborting scenario.")
                return

            if instruction_turn2:
                try:
                    turn1_result_image_pil = Image.open(turn1_output_path).convert("RGB")
                except FileNotFoundError:
                    print(f"     > Warning: {log_prefix} Turn1 result not found. Skipping turn2.")
                    return

                occluder_turn2_bbox = scenario_data.get("occluder_turn2_bbox")

                if (
                    not occluder_turn2_bbox
                    or not isinstance(occluder_turn2_bbox, list)
                    or len(occluder_turn2_bbox) != 4
                ):
                    print(f"     > Warning: {log_prefix} Invalid occluder_turn2_bbox. Proceeding without bbox for turn2.")
                    image_for_turn2 = turn1_result_image_pil
                else:
                    image_for_turn2 = draw_bbox_on_image(
                        turn1_result_image_pil, occluder_turn2_bbox, color="green", width=5
                    )
                    print(f"     > {log_prefix} Drew occluder_turn2_bbox {occluder_turn2_bbox} on the turn1 image.")

                print(f"     > {log_prefix} Sending turn2: {instruction_turn2[:50]}...")
                response2 = chat.send_message([instruction_turn2, image_for_turn2])
                _save_first_image_from_response(response2, str(turn2_output_path))

            time.sleep(1)

        except Exception as e_scenario:
            print(f"     > {log_prefix} Error during scenario processing: {e_scenario}")
            return

        print(f"     > {log_prefix} Done.")

    except Exception as e_file:
        print(f" > {log_prefix} File processing error: {e_file}")
        return

    return filename


def run_gemini_pipeline(
    scenario_root: Path,
    original_img_root: Path,
    base_results_root: Path,
    model_id: str = "models/gemini-2.5-flash-image",
    max_workers: int = 8,
):
    """
    Full Gemini inference pipeline for NY-BENCH.
    """
    MAX_WORKERS = max_workers

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment.")
        return

    genai.configure(api_key=api_key)

    try:
        model_instance = genai.GenerativeModel(model_id)
    except Exception as e:
        print(f"Gemini model initialization failed: {e}")
        return

    results_dir = base_results_root / "Gemini"
    results_dir.mkdir(parents=True, exist_ok=True)

    scenario_files = sorted(
        f for f in os.listdir(scenario_root) if f.endswith(".json")
    )

    print(f"--- GEMINI Inference Pipeline ---")
    print(f"Total scenarios: {len(scenario_files)}")
    print(f"Threads: {MAX_WORKERS}")
    print(f"Output directory: {results_dir}")
    print(f"Model: {model_id}")

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for filename in scenario_files:
            futures.append(
                executor.submit(
                    _process_single_scenario,
                    filename,
                    original_img_root,
                    scenario_root,
                    results_dir,
                    model_instance,
                )
            )

    completed_count = 0
    total_count = len(futures)

    for future in concurrent.futures.as_completed(futures):
        try:
            result_filename = future.result()
            if result_filename:
                completed_count += 1
        except Exception as e:
            print(f"   [Error] Worker failed: {e}")

    print(f"\n--- GEMINI Pipeline Complete ---")
    print(f"Completed: {completed_count} / {total_count}")


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="NY-BENCH: Multi-turn editing with Gemini"
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
        default="models/gemini-2.5-flash-image",
        help="Gemini model ID (default: models/gemini-2.5-flash-image)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel threads (default: 8)",
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

    run_gemini_pipeline(
        scenario_root=scenario_root,
        original_img_root=original_img_root,
        base_results_root=pred_root,
        model_id=args.model_id,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
