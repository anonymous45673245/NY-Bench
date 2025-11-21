Here is a comprehensive and professional README.md file, written in English, reflecting all the installation steps, environment settings, and pipeline details we discussed.

You can save this as README.md in your project root.

🚀 NY-BENCH: A Comprehensive Benchmark for Controllable Image Editing
NY-BENCH is designed to evaluate the multi-turn, multi-object editing capabilities of image editing models. It focuses on two core metrics:

Visual Consistency (VC): How well the background and non-target objects are preserved.

Instruction Faithfulness (IF): How accurately the edit reflects the text instruction.

This repository contains the official evaluation pipeline, tools, and metric calculations.

📂 Directory Structure
Ensure your project is organized as follows before running the evaluation:

Plaintext

NY-BENCH/
├── data/                    # Dataset (Must be downloaded separately)
│   ├── images/              # Original and GT images
│   ├── scenarios/           # Scenario JSON files
│   └── metadata.csv
├── predictions/             # Your model's output images
│   └── <Model_Name>/        # e.g., Gemini, InstructPix2Pix
├── detections/              # Output of Step 1 (JSONs)
├── masks/                   # Output of Step 2 (Binary Masks)
├── models/                  # Model Checkpoints
│   └── sam/
│       └── sam_vit_h.pth    # SAM ViT-H Checkpoint
├── nybench/                 # Core Library Code
├── scripts/                 # Evaluation Scripts (02, 03, 04)
└── requirements.txt         # Dependencies
🛠️ Installation & Prerequisites
1. Environment Setup (Linux Recommended)
We recommend using Conda to manage dependencies.

Bash

# 1. Create and activate a new environment (Python 3.10 recommended)
conda create -n nybench python=3.10 -y
conda activate nybench

# 2. Install core dependencies
pip install -r requirements.txt
2. Install Segment Anything (SAM)
The masking pipeline relies on the official SAM implementation. Install it directly from GitHub:

Bash

pip install "git+https://github.com/facebookresearch/segment-anything.git"
3. Set DeepDataSpace (DDS) Token
The detection step uses GroundingDINO via the DeepDataSpace API. You must obtain a token and export it as an environment variable.

Go to the DeepDataSpace Website.

Register/Login and get your API Token.

Export it in your terminal (ensure no trailing comments or spaces):

Bash

export DDS_TOKEN="your_token_here_without_spaces"
4. Download Model Checkpoints
Download the SAM ViT-H checkpoint (required for the masking step).

Download Link: sam_vit_h_4b8939.pth

Placement: Move the file to models/sam/ (or update the path in the script arguments later).

🏃 Running the Evaluation Pipeline
The evaluation is a sequential process. Output files from one step are required for the next.

Step 0: Prepare Predictions
Place your model's edited images in the predictions/ folder.

Path: predictions/<Model_Name>/

Naming Convention: <session_id>_<model_name>_turn<1|2>.png

Example: 0001_s3_Gemini_turn1.png

Step 1: Object Detection
Uses GroundingDINO to identify the bounding boxes of the objects mentioned in the editing instructions.

Bash

# Run detection (requires valid DDS_TOKEN)
python scripts/02_run_detection.py \
    --models Gemini \
    --api_token $DDS_TOKEN
Output: JSON files created in detections/<Model_Name>/.

Step 2: Mask Generation
Uses SAM to generate precise binary masks based on the bounding boxes from Step 1.

Bash

# Run masking (Requires SAM checkpoint)
python scripts/03_run_masking.py \
    --models Gemini \
    --sam-ckpt models/sam/sam_vit_h_4b8939.pth \
    --model-type vit_h
Output: Mask PNGs created in masks/<Model_Name>/ and JSONs updated.

Step 3: Final Evaluation
Calculates the Visual Consistency (VC) using LPIPS/PSNR and aggregates it with the Instruction Faithfulness (IF) scores.

Note: This step requires a pre-computed CSV containing LLM-based scores (data/LLM_IF.csv). If you do not have API access, ensure the CSV is populated manually or via a separate process.

Bash

# Run final scoring
python scripts/04_eval_full_final_score.py \
    --models Gemini \
    --llm_csv data/LLM_IF.csv
Output: * final_results.csv (Per-session breakdown)

final_summary.csv (Aggregated scores per model)

📚 Library Overview (nybench/)
nybench/schemas.py: Defines the standardized JSON structure used across the pipeline.

nybench/paths.py: Centralized path management utility.

nybench/detection/: Wrappers for GroundingDINO and SAM.

nybench/metrics/: Core logic for calculating VC (LPIPS, PSNR) and IF (Gemini API) metrics.

📝 License
[Insert License Information Here]