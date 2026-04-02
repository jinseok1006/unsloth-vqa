# AGENTS.md

## Project Overview

- This repository is for fine-tuning a VQA model.
- The primary working environment is `Google Colab Pro+` with Python `3.11`.
- Assume the default runtime is `GPU: A100` with `High-RAM` enabled.

## Main File

- The main working file is `jinseok.py`.
- Use `jinseok.py` as the primary source of truth for notebook-style training and experimentation changes.
- Use `train_unsloth_qwen35_9b_colab.py` as the reference implementation for the Unsloth training flow.

## Notebook Sync Rule

- There is a paired `.ipynb` notebook managed by `jupytext`.
- Always edit the `.py` file, not the `.ipynb` file.
- Changes to `jinseok.py` are expected to sync to the notebook automatically.

## Working Guidelines

- Prefer minimal, targeted edits over broad refactors.
- Preserve the existing notebook-style `# %%` cell structure unless there is a clear reason to change it.
- Keep code compatible with Colab Python `3.11`.
- Assume GPU execution on Colab `A100 + High-RAM` unless the task explicitly requires a fallback.
- When adding environment setup, prefer Colab notebook cells and pip-based installation instructions over local-only tooling.

## Data And Storage Defaults

- The default data workflow is: upload zipped image assets to Google Drive, unzip them inside Drive, then use those extracted assets from Colab.
- Assume `train.csv`, `test.csv`, `sample_submission.csv`, `train/`, and `test/` live under one dataset root in Drive after extraction.
- Prefer keeping the source dataset in Drive and staging copies into `/content` only when faster local I/O is needed.
- Persist outputs such as checkpoints, logs, validation files, and submissions to a Drive-backed output directory.

## Practical Defaults

- Prioritize changes related to VQA fine-tuning workflow, data loading, training, inference, and evaluation in the main script.
- For environment setup, use `(260324)_baseline_colab.ipynb` as the reference for Colab-oriented installation and runtime initialization.
- For Unsloth-specific loading and trainer wiring, refer to `DAY13_효율적인_Fine_tuning_PEFT_sol.ipynb` and `train_unsloth_qwen35_9b_colab.py`.
- Avoid manually editing generated or synchronized notebook artifacts.
- If both notebook and script appear out of sync, reconcile changes through the `.py` file first.
