# nybench/paths.py

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class ScenarioPaths:
    """
    Container for holding all file paths associated with a single NY-BENCH scenario.
    """
    scenario_id: str
    image_id: str
    scenario_idx: str
    turn0: Path
    gt_turn1: Path
    gt_turn2: Path
    scenario_json: Path


class NYBenchPaths:
    """
    Utility class for constructing file paths in the NY-BENCH dataset.
    The data_root should point to the directory 'NY-BENCH/data'.
    """

    def __init__(self, data_root: Union[str, Path, None] = None):
        """
        Args:
            data_root: Path to the 'data' directory. 
                       If None, tries to infer from file location.
        """
        if data_root:
            self.data_root = Path(data_root).resolve()
        else:
            # Fallback: assume this file is in NY-BENCH/nybench/paths.py
            # So project_root is two levels up, then into data
            self.data_root = Path(__file__).resolve().parent.parent / "data"

        self.project_root = self.data_root.parent

        # --- Standard Data Paths (Original) ---
        self.images_root = self.data_root / "images"
        self.scenarios_root = self.data_root / "scenarios"

        # --- Pipeline Output Paths (New) ---
        self.predictions_root = self.project_root / "predictions"
        self.detections_root = self.project_root / "detections"
        self.masks_root = self.project_root / "masks"

    def get_scenario_paths(
        self,
        scenario_id: str,
        image_id: str,
        scenario_idx: str,
    ) -> ScenarioPaths:
        """
        Given scenario identifiers, return all relevant image and JSON paths
        for that scenario.
        """
        img_dir = self.images_root / scenario_id
        turn0 = img_dir / "turn0.png"
        gt_turn1 = img_dir / "gt_turn1.png"
        gt_turn2 = img_dir / "gt_turn2.png"

        scenario_json = self.scenarios_root / f"{scenario_id}.json"

        return ScenarioPaths(
            scenario_id=scenario_id,
            image_id=image_id,
            scenario_idx=scenario_idx,
            turn0=turn0,
            gt_turn1=gt_turn1,
            gt_turn2=gt_turn2,
            scenario_json=scenario_json,
        )

    # --- New Methods for Detection & Masking ---

    def get_detection_dir(self, model_name: str) -> Path:
        """Returns path to NY-BENCH/detections/<model_name>/"""
        return self.detections_root / model_name

    def get_mask_dir(self, model_name: str) -> Path:
        """Returns path to NY-BENCH/masks/<model_name>/"""
        return self.masks_root / model_name

    def get_prediction_dir(self, model_name: str) -> Path:
        """Returns path to NY-BENCH/predictions/<model_name>/"""
        return self.predictions_root / model_name

    def get_prediction_image_path(self, model_name: str, session_id: str, turn_idx: int) -> Path:
        """
        Constructs the expected path for a prediction image.
        Format: predictions/<model>/<session_id>_<model>_turn<turn>.png
        """
        filename = f"{session_id}_{model_name}_turn{turn_idx}.png"
        return self.get_prediction_dir(model_name) / filename
