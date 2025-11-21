#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/04_eval_full_final_score.py

The final evaluation script for NY-BENCH.
Updated to correctly calculate S_VC (Visual Consistency) as the AVERAGE of Restore and Preserve scores.
Formula: S_VC = (S_restore + S_preserve) / 2
"""

import argparse
import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# --- Import Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from nybench.paths import NYBenchPaths
from nybench.metrics.visual_consistency import VisualConsistencyMetric
from nybench.metrics.instruction_faithfulness import InstructionFaithfulnessMetric

# Default location for the LLM score CSV
DEFAULT_LLM_CSV = "data/LLM_IF.csv"


def main():
    parser = argparse.ArgumentParser(description="NY-BENCH Final Evaluation")
    parser.add_argument("--models", nargs="+", help="List of models to evaluate")
    parser.add_argument("--llm_csv", type=str, default=DEFAULT_LLM_CSV, help="Path to LLM_IF.csv")
    parser.add_argument("--device", type=str, default="cuda", help="Device for VC metrics")
    args = parser.parse_args()

    # 1. Initialize System
    paths = NYBenchPaths()
    
    print("--- Initializing Metrics ---")
    # VC Metric (Calculates LPIPS/PSNR)
    vc_metric = VisualConsistencyMetric(device=args.device)
    
    # IF Metric (Loads CSV lookup)
    llm_csv_path = PROJECT_ROOT / args.llm_csv
    if_metric = InstructionFaithfulnessMetric(csv_path=llm_csv_path)
    print(f"Loaded IF scores from: {llm_csv_path}")

    # 2. Determine Models to Process
    detections_root = paths.get_detection_dir("") # Get base detection dir
    # Handle the case where detections_root points to the base 'detections' folder
    if detections_root.name != "detections": 
         detections_root = paths.detections_root

    if args.models:
        model_dirs = [detections_root / m for m in args.models]
    else:
        model_dirs = [d for d in detections_root.iterdir() if d.is_dir()]

    all_results = []

    # 3. Evaluation Loop
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\n=== Evaluating Model: {model_name} ===")
        
        # List all sessions (JSONs)
        json_files = sorted(list(model_dir.glob("*.json")))
        if not json_files:
            print(f"[WARN] No JSON files found in {model_dir}")
            continue

        for json_file in tqdm(json_files, desc=f"Scoring {model_name}"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load {json_file}: {e}")
                continue
            
            session_id = data.get('session_id')
            category = data.get('category', 'unknown')
            
            # --- A. Instruction Faithfulness (IF) ---
            # Get normalized score (Turn1 + Turn2) / 10
            # S_IF in the formula
            if_sum_norm = if_metric.get_normalized_session_score(session_id, model_name)
            score_if_t1 = if_metric.get_score(session_id, model_name, 1)
            score_if_t2 = if_metric.get_score(session_id, model_name, 2)

            # --- B. Visual Consistency (VC) ---
            # Compare Edited Image (Turn X) vs Original Image (Turn 0)
            turn0_img_path = paths.images_root / session_id / "turn0.png"
            
            restore_scores = []
            preserve_scores = []
            
            for turn in data.get('turns', []):
                turn_idx = turn.get('turn_index')
                if turn_idx == 0: continue

                # 1. Image Path
                pred_img_path = paths.get_prediction_image_path(model_name, session_id, turn_idx)
                
                # 2. Mask Path
                # Use mask_path from JSON (it is stored as relative path by 03_run_masking.py)
                mask_rel_path = None
                if turn.get('occluders'):
                    mask_rel_path = turn['occluders'][0].get('mask_path')
                
                full_mask_path = None
                if mask_rel_path:
                    full_mask_path = PROJECT_ROOT / mask_rel_path
                
                # Validate paths
                if not pred_img_path.exists():
                    # Try fallback if JSON has absolute path from older runs
                    if Path(turn.get('image_path', '')).exists():
                        pred_img_path = Path(turn['image_path'])
                    else:
                        continue # Cannot compute VC without image

                # Compute VC for this turn
                scores = vc_metric.calculate_scores(
                    img_ref_path=str(turn0_img_path),
                    img_curr_path=str(pred_img_path),
                    mask_fg_path=str(full_mask_path) if full_mask_path and full_mask_path.exists() else None
                )
                
                restore_scores.append(scores['S_restore'])
                preserve_scores.append(scores['S_preserve'])

            # Average VC across turns (Turn 1 & Turn 2)
            s_restore = np.mean(restore_scores) if restore_scores else 0.0
            s_preserve = np.mean(preserve_scores) if preserve_scores else 0.0
            
            # --- C. Final Score Calculation ---
            
            # [FIXED HERE] S_VC = (S_restore + S_preserve) / 2
            # Based on the formula: S_VC = (VC_restore + VC_preserve) / 2
            score_vc = (s_restore + s_preserve) / 2.0
            
            # Final Score = S_IF * S_VC
            final_score = if_sum_norm * score_vc
            
            all_results.append({
                'session_id': session_id,
                'model': model_name,
                'category': category,
                'Score_IF_t1': score_if_t1,
                'Score_IF_t2': score_if_t2,
                'IF_sum_norm': if_sum_norm,     # S_IF
                'S_restore': s_restore,         # VC_restore
                'S_preserve': s_preserve,       # VC_preserve
                'Score_UC_add': score_vc,       # S_VC (This column now contains the Average)
                'Final_Score': final_score
            })

    # 4. Save Results
    if not all_results:
        print("No results to save.")
        return

    df = pd.DataFrame(all_results)
    
    # Save Detailed
    out_csv = PROJECT_ROOT / "final_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nDetailed results saved to: {out_csv}")
    
    # Save Summary
    summary = df.groupby('model').agg({
        'Final_Score': 'mean',
        'IF_sum_norm': 'mean',
        'Score_UC_add': 'mean',
        'S_restore': 'mean',
        'S_preserve': 'mean',
        'session_id': 'count'
    }).rename(columns={'session_id': 'count'})
    
    summary_csv = PROJECT_ROOT / "final_summary.csv"
    summary.to_csv(summary_csv)
    
    print("\n=== Final Summary ===")
    print(summary)
    print(f"Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()