# nybench/detection/sam_wrapper.py
"""
SAM (Segment Anything Model) wrapper for NY-BENCH.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def resolve_sam_path(cli_sam_dir: Optional[str] = None) -> str:
    """
    Resolve the SAM code directory.
    """
    candidates = []
    if cli_sam_dir:
        candidates.append(cli_sam_dir)

    # [Modified for Anonymity] Removed hardcoded absolute paths.
    # Only use relative paths or CLI arguments.
    
    # Relative path fallback (assumes models/sam is inside the project)
    here = Path(__file__).resolve()
    rel_sam = here.parents[2] / "models" / "sam"
    candidates.append(str(rel_sam))

    for p in candidates:
        if p and Path(p).exists():
            return p
    return ""


def ensure_import_segment_anything(sam_dir: str):
    """
    Ensure that segment_anything package can be imported.
    """
    try:
        import segment_anything  # noqa: F401
        return
    except Exception:
        pass

    if sam_dir and (Path(sam_dir) / "segment_anything").exists():
        if sam_dir not in sys.path:
            sys.path.insert(0, sam_dir)
        try:
            import segment_anything  # noqa: F401
            return
        except Exception as e:
            raise ImportError(
                f"Failed to import segment_anything: {e}\n"
                f"Checked path: {sam_dir}\n"
            )
    else:
        # Just try normally, maybe installed via pip
        try:
            import segment_anything
        except ImportError:
            raise ImportError(
                "Failed to import segment_anything.\n"
                f"Tried path: '{sam_dir or '(none)'}'\n"
            )


def parse_box(
    box: Tuple[float, float, float, float],
    as_pixels: bool,
    w: int,
    h: int,
) -> np.ndarray:
    """
    Convert bbox to pixel coordinates.
    box: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = map(float, box)
    if not as_pixels:
        pass

    # Force recalculation for normalized input
    if not as_pixels:
        x1 = x1 * w
        y1 = y1 * h
        x2 = x2 * w
        y2 = y2 * h
    
    return np.array([x1, y1, x2, y2], dtype=np.float32)
