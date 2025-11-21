import os
import json
import time
import logging
import pandas as pd
import google.generativeai as genai
from google.api_core import retry
from pathlib import Path
from typing import Optional, Dict, Any, Union
from PIL import Image

# --- Configuration ---
DEFAULT_MODEL_VERSION = "gemini-2.5-pro" 

# --- Prompt Template (Gemini Output Format) ---
# Note: The LLM still outputs JSON with keys like "final_compliance".
# We will map these to "Score_IC" when saving to CSV.
JUDGE_PROMPT_TEMPLATE = """
Your sole task is to act as an image-editing evaluator.
You will be given two images and one instruction.
- The first image is the 'original image' (before the edit).
- The second image is the 'edited image' (after the edit).

You must only return a single, valid JSON object.

---
### Evaluation Task Details

Based on the provided 'original image', 'edited image', and 'instruction', evaluate the following items.

**Bounding box rule:**
- This is turn {TURN_INDEX}.
- In turn 1, the primary target box is RED. An optional secondary box (BLUE) may also be present if mentioned in the instruction.
- In turn 2, the target box is GREEN. Exception: 'Remove' instructions do not have a GREEN box (as they refer to the object from turn 1).
- Always use the box color explicitly mentioned in the instruction as the reference region when judging location and size.

**Metadata:**
- session_id: {SESSION_ID}
- turn_index: {TURN_INDEX}
- model: {MODEL_NAME}

**Instruction:**
{INSTRUCTION}

---
### Evaluation Criteria

Please rate the following questions on a scale of 1-5.

1) Location Accuracy (1-5)
Rate how accurately the change occurs at or near the target bounding box region.
(1 = completely wrong place, 5 = perfectly within/around the target box.)

2) Object Correctness (1-5)
Rate whether the correct type, color, and object were modified according to the instruction.
(1 = wrong object/action, 5 = perfectly correct.)

3) Action-Specific Size Change (1-5)
Rate how well the size change matches the intended action:
- Add -> new object appears with reasonable size
- Remove -> object disappears
- Move -> the same object is moved, with a consistent size
- Replace / Shrink -> Follow the size intent:
  If the instruction explicitly mentions a size change (e.g., "smaller teddy bear", "shrink the balloon"), follow that description.
  If no size change is mentioned, infer the expected size behavior from the bounding box design:
  for example, if the new (green) box is smaller than the previous one, the resulting object should appear smaller accordingly.
(1 = wrong size behavior, 5 = correct and natural size change.)

4) Final Compliance Score (1-5)
Give an overall score summarizing how faithfully the edit follows the instruction.

---
### Output Format

**Your response must *only* be in the following JSON format. Do NOT include markdown (```json ... ```) or any other text.**
{{
  "session_id": "{SESSION_ID}",
  "turn_index": {TURN_INDEX},
  "model": "{MODEL_NAME}",
  "location_accuracy": <integer 1-5>,
  "object_correctness": <integer 1-5>,
  "action_size_change": <integer 1-5>,
  "final_compliance": <integer 1-5>,
  "final_reason": "<one concise sentence explaining the final score>"
}}
"""

class InstructionFaithfulnessMetric:
    """
    Calculates Instruction Faithfulness (IF) scores using LLM-as-a-Judge (Gemini).
    Adapts to the specific CSV format: 
    [session_id, turn_index, model_type, category, S_modify, S_position, S_size, Score_IC]
    """

    def __init__(self, csv_path: Union[str, Path] = "data/LLM_IF.csv", api_key: Optional[str] = None, model_version: str = DEFAULT_MODEL_VERSION):
        self.logger = logging.getLogger(__name__)
        self.csv_path = Path(csv_path)
        self.model_version = model_version
        
        # 1. Setup Gemini API
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_version)
        else:
            self.logger.warning("[IF Metric] No GEMINI_API_KEY found. Only cached scores will be available.")

        # 2. Load Cache
        self.cache_df = pd.DataFrame()
        self._load_cache()

    def _load_cache(self):
        """Loads the CSV file into memory."""
        if self.csv_path.exists():
            try:
                self.cache_df = pd.read_csv(self.csv_path)
                
                # Standardize column types for querying
                if not self.cache_df.empty:
                    # Ensure keys exist
                    if 'session_id' in self.cache_df.columns:
                        self.cache_df['session_id'] = self.cache_df['session_id'].astype(str)
                    if 'model_type' in self.cache_df.columns:
                        self.cache_df['model_type'] = self.cache_df['model_type'].astype(str)
                    if 'turn_index' in self.cache_df.columns:
                        self.cache_df['turn_index'] = self.cache_df['turn_index'].astype(int)
                    
                self.logger.info(f"[IF Metric] Loaded {len(self.cache_df)} scores from {self.csv_path}")
            except Exception as e:
                self.logger.error(f"[IF Metric] Failed to load CSV: {e}")
                self._init_empty_cache()
        else:
            self._init_empty_cache()

    def _init_empty_cache(self):
        """Initialize DataFrame with the specific columns requested."""
        self.cache_df = pd.DataFrame(columns=[
            'session_id', 'turn_index', 'model_type', 'category',
            'S_modify', 'S_position', 'S_size', 'Score_IC'
        ])

    def _save_cache(self):
        """Saves the current dataframe to CSV."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_df.to_csv(self.csv_path, index=False)

    def _check_cache(self, session_id: str, model_name: str, turn_index: int) -> Optional[float]:
        """
        Returns 'Score_IC' if exists, else None.
        Uses 'model_type' column for model name.
        """
        if self.cache_df.empty:
            return None
            
        # Check for required columns just in case
        if 'model_type' not in self.cache_df.columns or 'Score_IC' not in self.cache_df.columns:
            return None

        match = self.cache_df[
            (self.cache_df['session_id'] == str(session_id)) &
            (self.cache_df['model_type'] == str(model_name)) &
            (self.cache_df['turn_index'] == int(turn_index))
        ]
        
        if not match.empty:
            val = match.iloc[0]['Score_IC']
            return float(val)
        return None

    @retry.Retry(predicate=retry.if_exception_type(Exception))
    def _call_api(self, img_ref_path: Path, img_curr_path: Path, prompt: str) -> Dict[str, Any]:
        """Internal method to call Gemini API."""
        if not self.model:
            raise ValueError("Gemini model not initialized.")

        img0 = Image.open(img_ref_path)
        img1 = Image.open(img_curr_path)

        response = self.model.generate_content([prompt, img0, img1])
        
        try:
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            self.logger.error(f"[IF Metric] API Response Parse Error: {response.text}")
            raise e

    def get_score(self, 
                  session_id: str, 
                  model_name: str, 
                  turn_index: int, 
                  instruction: Optional[str] = None,
                  img_ref_path: Optional[Union[str, Path]] = None,
                  img_curr_path: Optional[Union[str, Path]] = None) -> float:
        """
        Retrieves Score_IC (Instruction Compliance).
        If not in cache and inputs provided, computes via Gemini and saves to CSV.
        """
        # 1. Check Cache
        cached_score = self._check_cache(session_id, model_name, turn_index)
        if cached_score is not None:
            return cached_score

        # 2. If Read-Only (no inputs), return 0
        if not all([instruction, img_ref_path, img_curr_path]):
            return 0.0

        # 3. Compute via API
        self.logger.info(f"[IF Metric] Computing score for {session_id} | {model_name} | T{turn_index}")
        
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            TURN_INDEX=turn_index,
            SESSION_ID=session_id,
            MODEL_NAME=model_name,
            INSTRUCTION=instruction
        )
        
        try:
            # Get JSON from LLM
            result = self._call_api(Path(img_ref_path), Path(img_curr_path), prompt)
            final_score = float(result.get('final_compliance', 0))
            
            # Map LLM JSON keys to CSV Column names
            # JSON: object_correctness -> CSV: S_modify
            # JSON: location_accuracy  -> CSV: S_position
            # JSON: action_size_change -> CSV: S_size
            # JSON: final_compliance   -> CSV: Score_IC
            
            new_row = {
                'session_id': str(session_id),
                'turn_index': int(turn_index),
                'model_type': str(model_name),      # Mapped to model_type
                'category': 'unknown',              # Placeholder if not provided
                'S_position': result.get('location_accuracy', 0),
                'S_modify': result.get('object_correctness', 0),
                'S_size': result.get('action_size_change', 0),
                'Score_IC': final_score             # Mapped to Score_IC
            }
            
            self.cache_df = pd.concat([self.cache_df, pd.DataFrame([new_row])], ignore_index=True)
            self._save_cache()
            
            time.sleep(1)
            return final_score

        except Exception as e:
            self.logger.error(f"[IF Metric] Calculation failed: {e}")
            return 0.0

    def get_normalized_session_score(self, session_id: str, model_name: str) -> float:
        """
        Calculates (Turn1_Score_IC + Turn2_Score_IC) / 10.
        Assumes Score_IC is on a 1-5 scale.
        """
        s1 = self._check_cache(session_id, model_name, 1) or 0.0
        s2 = self._check_cache(session_id, model_name, 2) or 0.0
        
        if s1 == 0 and s2 == 0:
            return 0.0
            
        return (s1 + s2) / 10.0