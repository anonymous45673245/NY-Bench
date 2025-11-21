# nybench/dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset  # can be removed if not using torch

from .paths import NYBenchPaths, ScenarioPaths


class NYBenchDataset(Dataset):
    """
    Dataset class for NY-BENCH, returning one scenario per item.

    Args:
        data_root (str | Path): Path to NY-BENCH/data directory.
        split (str): Split name in metadata.csv (usually 'eval').
        load_images (bool): If True, loads PIL.Image objects.
                            If False, returns only file paths.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "eval",
        load_images: bool = False,
    ):
        self.data_root = Path(data_root)
        self.paths = NYBenchPaths(self.data_root)
        self.metadata_path = self.data_root / "metadata.csv"
        self.load_images = load_images

        # metadata.csv must exist
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata.csv not found at {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)

        # Filter by split column if exists
        if "split" in df.columns:
            df = df[df["split"] == split]

        # Required columns for dataset construction
        required_cols = ["scenario_id", "image_id", "scenario_idx"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"metadata.csv missing required columns: {missing}")

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: Path) -> Image.Image:
        """Load an RGB image using PIL."""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one scenario sample as a dictionary."""
        row = self.df.iloc[idx]
        scenario_id = str(row["scenario_id"])
        image_id = str(row["image_id"])
        scenario_idx = str(row["scenario_idx"])

        # Fetch all file paths associated with this scenario
        paths: ScenarioPaths = self.paths.get_scenario_paths(
            scenario_id=scenario_id,
            image_id=image_id,
            scenario_idx=scenario_idx,
        )

        # Base sample information
        sample: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "image_id": image_id,
            "scenario_idx": scenario_idx,
            "category": row.get("category", None),
            "source": row.get("source", None),
            "split": row.get("split", None),
            "paths": {
                "turn0": paths.turn0,
                "gt_turn1": paths.gt_turn1,
                "gt_turn2": paths.gt_turn2,
                "scenario_json": paths.scenario_json,
            },
        }

        # Optional: load actual image data
        if self.load_images:
            sample["images"] = {
                "turn0": self._load_image(paths.turn0),
                "gt_turn1": self._load_image(paths.gt_turn1),
                "gt_turn2": self._load_image(paths.gt_turn2),
            }

        return sample
