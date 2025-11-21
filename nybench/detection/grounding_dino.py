# nybench/detection/grounding_dino.py

from __future__ import annotations

import os
import time
import base64
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# DeepDataSpace API Endpoints
API_CREATE = "https://api.deepdataspace.com/v2/task/grounding_dino/detection"
API_URL_QUERY = "https://api.deepdataspace.com/v2/task_status/"
POLL_INTERVAL_S = 5  # seconds between polling attempts


@dataclass
class DetectionResult:
    """
    Container for API detection results.
    """
    objects: List[Dict[str, Any]]
    best_bbox: List[float]      # [x1, y1, x2, y2] or [0,0,0,0] if not found
    best_score: float           # score of best bbox, or 0.0
    raw_response: Dict[str, Any]


def encode_image_to_base64_uri(image_path: str | Path) -> str:
    """
    Encode an image file into a Base64 data URI string.
    """
    path = Path(image_path)
    ext = path.suffix.lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{encoded}"


def _poll_task_status(task_uuid: str, api_token: str) -> Dict[str, Any]:
    """
    Poll the task status from DeepDataSpace API until completion.
    Returns the final JSON response once the task succeeds.

    Raises:
        RuntimeError: if the task fails or returns an error state.
    """
    query_url = f"{API_URL_QUERY}{task_uuid}"
    headers = {"Token": api_token}

    while True:
        resp = requests.get(query_url, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        status = data.get("data", {}).get("status")

        if status == "success":
            return data
        if status in ("failed", "error"):
            raise RuntimeError(f"GroundingDINO task failed: {data}")

        # Not finished yet → wait and poll again
        time.sleep(POLL_INTERVAL_S)


def run_grounding_dino(
    image_path: str | Path,
    text: str,
    api_token: Optional[str] = None,
    model_name: str = "GroundingDino-1.6-Pro",
    bbox_threshold: float = 0.25,
    iou_threshold: float = 0.8,
    min_score: float = 0.3,
) -> DetectionResult:
    """
    Run GroundingDINO detection via DeepDataSpace API for a single
    image–text pair.

    Args:
        image_path: Path to the input image.
        text: Text prompt for grounding detection.
        api_token: DDS API token (or read from $DDS_TOKEN).
        model_name: GroundingDINO variant to use.
        bbox_threshold: Minimum bbox score for filtering.
        iou_threshold: IoU threshold for box merging.
        min_score: Score threshold to accept best bbox.

    Returns:
        DetectionResult object containing:
            - all detected objects
            - best bbox (raw pixel coords)
            - best score
            - raw API response
    """
    token = api_token or os.environ.get("DDS_TOKEN")
    if not token:
        raise RuntimeError("DDS_TOKEN environment variable or api_token argument is required.")

    headers = {"Token": token, "Content-Type": "application/json"}

    encoded_image = encode_image_to_base64_uri(image_path)

    payload = {
        "model": model_name,
        "image": encoded_image,
        "prompt": {"type": "text", "text": text},
        "targets": ["bbox"],
        "bbox_threshold": float(bbox_threshold),
        "iou_threshold": float(iou_threshold),
    }

    # Step 1: Create task
    resp = requests.post(API_CREATE, headers=headers, json=payload)
    resp.raise_for_status()

    task_data = resp.json()
    if task_data.get("code", 0) != 0:
        raise RuntimeError(f"GroundingDINO API error: {task_data.get('msg')}")

    task_uuid = task_data.get("data", {}).get("task_uuid")
    if not task_uuid:
        raise RuntimeError(f"task_uuid missing in response: {task_data}")

    # Step 2: Poll for result
    poll_data = _poll_task_status(task_uuid, token)

    objects = poll_data.get("data", {}).get("result", {}).get("objects", []) or []

    # No detections found
    if not objects:
        return DetectionResult(
            objects=[],
            best_bbox=[0.0, 0.0, 0.0, 0.0],
            best_score=0.0,
            raw_response=poll_data,
        )

    # Pick best by raw score
    candidates = [o for o in objects if isinstance(o.get("score"), (int, float))]
    best = max(candidates, key=lambda o: o["score"]) if candidates else None

    # Not valid or below threshold
    if best is None or best.get("score", 0.0) < float(min_score):
        return DetectionResult(
            objects=objects,
            best_bbox=[0.0, 0.0, 0.0, 0.0],
            best_score=0.0,
            raw_response=poll_data,
        )

    return DetectionResult(
        objects=objects,
        best_bbox=list(best.get("bbox", [0.0, 0.0, 0.0, 0.0])),
        best_score=float(best.get("score", 0.0)),
        raw_response=poll_data,
    )


def draw_single_bbox(
    image_path: str | Path,
    bbox: List[float],
    label: str,
    score: float,
    out_path: str | Path,
):
    """
    Draw a single bounding box on the image and save to disk.

    Args:
        image_path: Path to source image.
        bbox: Pixel coordinates [x1,y1,x2,y2].
        label: Label to draw.
        score: Confidence score to display.
        out_path: Output path for annotated image.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try loading a standard font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
        )
    except Exception:
        font = ImageFont.load_default()

    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    text = f"{label}: {score:.2f}"
    text_pos = (x1 + 2, y1 + 2)

    tb = draw.textbbox(text_pos, text, font=font)
    draw.rectangle(tb, fill="red")
    draw.text(text_pos, text, fill="white", font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
