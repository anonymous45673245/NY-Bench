#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_run_detection.py

Automated detection pipeline for NY-BENCH:

Repository layout (relative to project root):
    NY-BENCH/
    ├── data/
    │   ├── images/<session_id>/turn0.png, gt_turn1.png, gt_turn2.png
    │   ├── scenarios/<session_id>.json
    │   └── metadata.csv
    ├── predictions/<model_name>/*.png
    ├── detections/<model_name>/*.json
    └── masks/<model_name>/*.png  (used by 03_run_masking.py)

This script:
  - Scans prediction images in predictions/{model_name}/
  - Loads occluder text from data/scenarios/{session_id}.json
  - Calls GroundingDINO API via nybench.detection.grounding_dino
  - Saves detection JSONs to detections/{model_name}/
  - Logs low-confidence detections and errors
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from PIL import Image
import sys

# ----------------------------------------------------------------------
# Make project root importable so that `import nybench` works
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # NY-Bench/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from nybench.detection.grounding_dino import run_grounding_dino, DetectionResult


# ============================================================================
# Configuration
# ============================================================================

DEFAULTS = {
    # Path to `data/` directory (repo root is assumed as parent of this)
    
    "data_root": str(PROJECT_ROOT / "data"),

    # Detection hyper-parameters
    "bbox_threshold": 0.25,
    "iou_threshold": 0.8,
    "min_score": 0.3,
}

# Filename pattern: 0001_s0_GPT_turn1.png
FNAME_PATTERN = re.compile(
    r"^(?P<id>\d{4})_s(?P<scene>\d+)_(?P<model>[A-Za-z0-9\-]+)_turn(?P<turn>\d+)\.(?:png|jpg|jpeg)$",
    re.IGNORECASE,
)


# ============================================================================
# Utilities
# ============================================================================

def load_scenario(scenarios_dir: Path, session_id: str) -> Dict:
    """Load scenario JSON file for a given session."""
    jpath = scenarios_dir / f"{session_id}.json"
    with open(jpath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_occluder_text(scenario: Dict, scene: str, turn: int) -> Tuple[str, str]:
    """
    Extract occluder text from scenario.

    Returns:
        (occluder_text, occluder_star_text)
        occluder_star_text is only used in scene 5 (secondary occluder).
    """
    occluder = None
    occluder_star = None

    if turn == 1:
        occluder = scenario.get("occluder_turn1")
        if scene == "5":
            occluder_star = scenario.get("occluder*_turn1")
    elif turn == 2:
        occluder = scenario.get("occluder_turn2")
        if scene == "5":
            occluder_star = scenario.get("occluder*_turn2")

        # Warning check for s4, s5
        if scene in ["4", "5"]:
            if occluder is not None:
                print(
                    f"[WARNING] {scenario.get('session_id', 'unknown')} "
                    f"s{scene} turn2: occluder_turn2 is not null! Check scenario."
                )

    return occluder or "", occluder_star or ""


def normalize_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """Normalize [x1, y1, x2, y2] bounding box coordinates to [0, 1] range."""
    if img_width == 0 or img_height == 0:
        return [0.0, 0.0, 0.0, 0.0]

    x1, y1, x2, y2 = bbox
    return [
        x1 / img_width,
        y1 / img_height,
        x2 / img_width,
        y2 / img_height,
    ]


def create_output_json(session_id: str, scene: str, model_name: str, scenario: Dict) -> Dict:
    """
    Create output JSON structure.

    Paths are stored relative to the project root:
        - data/images/...
        - predictions/<model_name>/...
    """
    image_type = scenario.get("image_type", "natural")
    category = scenario.get("category", "")
    id4 = session_id.split("_")[0]

    return {
        "session_id": session_id,
        "image_type": image_type,
        "model_type": model_name,
        "category": category,
        "tools": {
            "detector": "GroundingDino-1.6-Pro",
            "segmentor": "sam",
        },
        "turns": [
            {
                "turn_index": 0,
                "image_path": f"data/images/{id4}_s{scene}/turn0.png",
                "occluders": [
                    {
                        "occluder_id": 0,
                        "occluder_text": "none",
                        "bounding_box": None,
                        "mask_path": None,
                        "confidence": None,
                    },
                    {
                        "occluder_id": 1,
                        "occluder_text": "none",
                        "bounding_box": None,
                        "mask_path": None,
                        "confidence": None,
                    },
                ],
            },
            {
                "turn_index": 1,
                "image_path": (
                    f"predictions/{model_name}/{session_id}_{model_name}_turn1.png"
                ),
                "occluders": [
                    {
                        "occluder_id": 0,
                        "occluder_text": "none",
                        "bounding_box": None,
                        "mask_path": None,
                        "confidence": None,
                    },
                    {
                        "occluder_id": 1,
                        "occluder_text": "none",
                        "bounding_box": None,
                        "mask_path": None,
                        "confidence": None,
                    },
                ],
            },
            {
                "turn_index": 2,
                "image_path": (
                    f"predictions/{model_name}/{session_id}_{model_name}_turn2.png"
                ),
                "occluders": [
                    {
                        "occluder_id": 0,
                        "occluder_text": "none",
                        "bounding_box": None,
                        "mask_path": None,
                        "confidence": None,
                    },
                    {
                        "occluder_id": 1,
                        "occluder_text": "none",
                        "bounding_box": None,
                        "mask_path": None,
                        "confidence": None,
                    },
                ],
            },
        ],
    }


def ensure_dir(p: Path) -> Path:
    """Ensure directory exists and return it."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_low_confidence(
    log_path: Path,
    session_id: str,
    model_name: str,
    turn: int,
    occ_id: int,
    text: str,
    score: float,
    img_path: str,
) -> None:
    """Log low confidence detections."""
    ts = datetime.now().isoformat(timespec="seconds")
    line = (
        f"[{ts}] session={session_id}, model={model_name}, "
        f"turn={turn}, occ_id={occ_id}, "
        f"text={text!r}, score={score:.4f}, img={img_path}"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_grounding_error(
    log_path: Path,
    session_id: str,
    model_name: str,
    turn: int,
    occ_id: int,
    text: str,
    error_msg: str,
) -> None:
    """Log errors during GroundingDINO calls or I/O."""
    ts = datetime.now().isoformat(timespec="seconds")
    line = (
        f"[{ts}] session={session_id}, model={model_name}, "
        f"turn={turn}, occ_id={occ_id}, "
        f"text={text!r}, error={error_msg}"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================================
# Main Pipeline
# ============================================================================

def build_argparser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    p = argparse.ArgumentParser(
        description="NY-BENCH GroundingDINO detection pipeline"
    )
    p.add_argument(
        "--data_root",
        default=DEFAULTS["data_root"],
        help="Path to data/ directory (default: ./data)",
    )
    p.add_argument(
        "--bbox_threshold",
        type=float,
        default=DEFAULTS["bbox_threshold"],
        help="GroundingDINO bbox threshold",
    )
    p.add_argument(
        "--iou_threshold",
        type=float,
        default=DEFAULTS["iou_threshold"],
        help="GroundingDINO IoU threshold",
    )
    p.add_argument(
        "--min_score",
        type=float,
        default=DEFAULTS["min_score"],
        help="Minimum confidence score to save bbox",
    )
    p.add_argument(
        "--models",
        nargs="*",
        help="Specific models to process (default: all subdirs in predictions/)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing detection JSONs",
    )
    p.add_argument(
        "--api_token",
        type=str,
        default=None,
        help="DeepDataSpace API token (or use $DDS_TOKEN if omitted)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    # Resolve paths based on data_root
    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    # Repo root is assumed as parent of data_root
    repo_root = data_root.parent

    scenarios_dir = data_root / "scenarios"
    predictions_root = repo_root / "predictions"
    detections_root = ensure_dir(repo_root / "detections")

    # API token: CLI arg takes precedence; otherwise environment
    api_token = args.api_token or os.environ.get("DDS_TOKEN", None)

    # Discover model directories under predictions/
    if args.models:
        model_dirs = [
            predictions_root / m
            for m in args.models
            if (predictions_root / m).is_dir()
        ]
    else:
        if not predictions_root.is_dir():
            print(f"[ERROR] predictions directory not found: {predictions_root}")
            return
        model_dirs = [d for d in predictions_root.iterdir() if d.is_dir()]

    if not model_dirs:
        print(f"[ERROR] No model directories found in {predictions_root}")
        return

    model_dirs = sorted(model_dirs)

    total_ok = 0
    total_skip = 0
    total_fail = 0

    # ----------------------------------------------------------------------
    # Process each model
    # ----------------------------------------------------------------------
    for model_dir in model_dirs:
        model_name = model_dir.name
        print("\n" + "=" * 70)
        print(f"Processing model: {model_name}")
        print("=" * 70)

        out_dir = ensure_dir(detections_root / model_name)
        error_log_path = out_dir / "errors.txt"
        low_conf_log_path = out_dir / "low_confidence.txt"

        # Scan image files
        files = sorted(
            f
            for f in os.listdir(model_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if not files:
            print(f"[WARNING] No images found in {model_dir}")
            continue

        # Group files by session
        sessions: Dict[str, Dict[int, str]] = {}
        for fname in files:
            m = FNAME_PATTERN.match(fname)
            if not m:
                print(f"[SKIP] Invalid filename: {fname}")
                continue

            id4 = m.group("id")
            scene = m.group("scene")
            turn = int(m.group("turn"))

            session_id = f"{id4}_s{scene}"
            if session_id not in sessions:
                sessions[session_id] = {}
            sessions[session_id][turn] = fname

        # ------------------------------------------------------------------
        # Process each session
        # ------------------------------------------------------------------
        for session_id in sorted(sessions.keys()):
            turns = sessions[session_id]
            id4, scene_part = session_id.split("_s")
            scene = scene_part

            out_json_path = out_dir / f"{session_id}_{model_name}.json"

            # Skip if already exists
            if out_json_path.exists() and not args.force:
                print(f"[SKIP] Already exists: {out_json_path}")
                total_skip += 1
                continue

            print(f"\n--- Processing: {session_id} ---")

            # Load scenario
            try:
                scenario = load_scenario(scenarios_dir, session_id)
            except FileNotFoundError:
                msg = "Scenario file not found"
                print(f"[ERROR] {msg}: {session_id}.json")
                total_fail += 1
                log_grounding_error(
                    error_log_path,
                    session_id,
                    model_name,
                    turn=0,
                    occ_id=0,
                    text="N/A",
                    error_msg=msg,
                )
                continue
            except Exception as e:
                msg = f"Scenario loading failed: {e}"
                print(f"[ERROR] {msg}")
                total_fail += 1
                log_grounding_error(
                    error_log_path,
                    session_id,
                    model_name,
                    turn=0,
                    occ_id=0,
                    text="N/A",
                    error_msg=msg,
                )
                continue

            # Create output JSON template
            output_json = create_output_json(session_id, scene, model_name, scenario)

            # --------------------------------------------------------------
            # Process each prediction turn (1 and 2)
            # --------------------------------------------------------------
            for turn in [1, 2]:
                if turn not in turns:
                    # Some sessions may miss a turn2 prediction, etc.
                    continue

                fname = turns[turn]
                img_path = model_dir / fname

                # Load image and get dimensions
                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except Exception as e:
                    msg = f"Image loading failed: {e}"
                    print(f"[ERROR] Failed to load image {fname}: {e}")
                    log_grounding_error(
                        error_log_path,
                        session_id,
                        model_name,
                        turn=turn,
                        occ_id=0,
                        text="N/A",
                        error_msg=msg,
                    )
                    continue

                # Get occluder texts
                occluder_text, occluder_star_text = get_occluder_text(
                    scenario, scene, turn
                )

                # ------------------------------
                # occluder_id 0 (main occluder)
                # ------------------------------
                if occluder_text:
                    print(f"  Turn {turn}, occluder_id 0: {occluder_text!r}")

                    try:
                        result: DetectionResult = run_grounding_dino(
                            image_path=str(img_path),
                            text=occluder_text,
                            api_token=api_token,
                            model_name="GroundingDino-1.6-Pro",
                            bbox_threshold=args.bbox_threshold,
                            iou_threshold=args.iou_threshold,
                            min_score=args.min_score,
                        )

                        output_json["turns"][turn]["occluders"][0][
                            "occluder_text"
                        ] = occluder_text

                        if result.best_score >= args.min_score:
                            norm_bbox = normalize_bbox(
                                result.best_bbox, img_width, img_height
                            )
                            output_json["turns"][turn]["occluders"][0][
                                "bounding_box"
                            ] = norm_bbox
                            output_json["turns"][turn]["occluders"][0][
                                "confidence"
                            ] = round(result.best_score, 4)
                            print(
                                f"    -> bbox: {norm_bbox}, "
                                f"confidence: {result.best_score:.4f}"
                            )
                        else:
                            output_json["turns"][turn]["occluders"][0][
                                "confidence"
                            ] = round(result.best_score, 4)
                            print(
                                f"    -> score too low: {result.best_score:.4f} "
                                f"(bbox not saved)"
                            )

                            log_low_confidence(
                                low_conf_log_path,
                                session_id=session_id,
                                model_name=model_name,
                                turn=turn,
                                occ_id=0,
                                text=occluder_text,
                                score=result.best_score,
                                img_path=str(img_path),
                            )

                    except Exception as e:
                        msg = f"GroundingDINO failed: {e}"
                        print(f"    -> {msg}")
                        log_grounding_error(
                            error_log_path,
                            session_id,
                            model_name,
                            turn=turn,
                            occ_id=0,
                            text=occluder_text,
                            error_msg=str(e),
                        )

                # -----------------------------------------
                # occluder_id 1 (secondary, scene 5 only)
                # -----------------------------------------
                if occluder_star_text and scene == "5":
                    print(f"  Turn {turn}, occluder_id 1: {occluder_star_text!r}")

                    try:
                        result: DetectionResult = run_grounding_dino(
                            image_path=str(img_path),
                            text=occluder_star_text,
                            api_token=api_token,
                            model_name="GroundingDino-1.6-Pro",
                            bbox_threshold=args.bbox_threshold,
                            iou_threshold=args.iou_threshold,
                            min_score=args.min_score,
                        )

                        output_json["turns"][turn]["occluders"][1][
                            "occluder_text"
                        ] = occluder_star_text

                        if result.best_score >= args.min_score:
                            norm_bbox = normalize_bbox(
                                result.best_bbox, img_width, img_height
                            )
                            output_json["turns"][turn]["occluders"][1][
                                "bounding_box"
                            ] = norm_bbox
                            output_json["turns"][turn]["occluders"][1][
                                "confidence"
                            ] = round(result.best_score, 4)
                            print(
                                f"    -> bbox: {norm_bbox}, "
                                f"confidence: {result.best_score:.4f}"
                            )
                        else:
                            output_json["turns"][turn]["occluders"][1][
                                "confidence"
                            ] = round(result.best_score, 4)
                            print(
                                f"    -> score too low: {result.best_score:.4f} "
                                f"(bbox not saved)"
                            )

                            log_low_confidence(
                                low_conf_log_path,
                                session_id=session_id,
                                model_name=model_name,
                                turn=turn,
                                occ_id=1,
                                text=occluder_star_text,
                                score=result.best_score,
                                img_path=str(img_path),
                            )

                    except Exception as e:
                        msg = f"GroundingDINO failed: {e}"
                        print(f"    -> {msg}")
                        log_grounding_error(
                            error_log_path,
                            session_id,
                            model_name,
                            turn=turn,
                            occ_id=1,
                            text=occluder_star_text,
                            error_msg=str(e),
                        )

            # --------------------------------------------------------------
            # Save detection JSON for this (session, model)
            # --------------------------------------------------------------
            try:
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(output_json, f, ensure_ascii=False, indent=2)

                print(f"[OK] Saved: {out_json_path}")
                total_ok += 1

            except Exception as e:
                msg = f"JSON save failed: {e}"
                print(f"[ERROR] {msg}")
                log_grounding_error(
                    error_log_path,
                    session_id,
                    model_name,
                    turn=0,
                    occ_id=0,
                    text="N/A",
                    error_msg=msg,
                )
                total_fail += 1

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"[Summary] OK={total_ok} | SKIP={total_skip} | FAIL={total_fail}")
    print("=" * 70)


if __name__ == "__main__":
    main()
