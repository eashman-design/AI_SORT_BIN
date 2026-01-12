# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI_SORT_BIN is an automated waste sorting system that uses computer vision to classify objects and actuate physical bins. The system runs on Ubuntu 24.04 with an NVIDIA GTX 1080 GPU.

## Commands

```bash
# Verify environment (GPU, TensorFlow, OpenCV)
python smoke_test.py

# Run the sorting runtime (continuous loop)
python -m scripts.runtime.main

# Data preparation
python scripts/training/preprocess.py      # Resize images to 224x224
python scripts/training/dataset_check.py   # Validate dataset integrity
python scripts/training/gpu_test.py        # Test TensorFlow GPU setup
```

## Architecture

### Runtime Pipeline

The runtime uses a 5-state finite state machine that cycles continuously:

```
IDLE → DETECT → DECIDE → ACTUATE → RESET → (loop)
```

- **StateMachine** (`scripts/runtime/state_machine.py`): Orchestrates the detection→decision→actuation cycle
- **DecisionEngine** (`scripts/decision/decision_engine.py`): Policy-based bin selection using confidence thresholds from `bin_policy.yaml`
- **InferenceEngine**: Currently a stub; will run the ML model
- **Actuator**: Currently a stub; will control GPIO/servo hardware

### Configuration

`scripts/decision/bin_policy.yaml` defines:
- Bin ID mappings for each waste class (trash, containers, paper, glass)
- Per-class confidence thresholds
- Fallback bin for uncertain classifications

### Training Pipeline

Training utilities exist in `scripts/training/` but no model training script yet:
- `preprocess.py`: Standardizes images to 224×224 (for MobileNetV2)
- `augment.py`: Data augmentation functions (flip, brightness, contrast, saturation, JPEG noise)
- Target model input size: 224×224

### Dataset Structure

```
dataset/
├── raw/<class>/           # Original images
├── processed/<class>/     # Production preprocessed (train/valid/test splits)
└── processed_test/<class>/ # Development/test images
```

Classes: plastic, paper, metal, trash (defined in `dataset/dataset_config.json`)

## Environment Constraints

- Python 3.10 (conda env: `ai_sort_bin`)
- TensorFlow 2.15.1 with CUDA
- NumPy <2.0 (required for compatibility)
- OpenCV <4.12 (4.12+ forces NumPy 2)


# Claude Code – Project Instructions (AI_SORT_BIN)

## Project Context
This repository implements an AI-assisted waste-sorting system designed for
real-world deployment in a school environment. The system includes:
- A training pipeline (dataset prep, augmentation, model training)
- A runtime pipeline (inference, decision logic, actuator control)
- Strong emphasis on correctness, stability, and future extensibility

## Standing Rules (Apply to All Tasks)
- Treat this as a production research repository.
- Do not change runtime behavior unless explicitly instructed.
- Do not delete files or folders without asking first.
- Prefer minimal diffs over large rewrites.
- Maintain existing directory structure unless told otherwise.
- Avoid introducing new dependencies unless approved.
- Assume Python 3.12 compatibility is required.
- If instructions are ambiguous, stop and ask before proceeding.

## Workflow Expectations
- Summarize planned changes before modifying files.
- Make changes incrementally.
- Run tests or smoke checks when applicable.
- Provide a concise summary of changes after completion.

## Git Discipline
- Assume work is done on a feature branch unless stated otherwise.
- Do not squash or rewrite history unless instructed.

## Safety Principle
When uncertain, choose the safest action and request clarification.



