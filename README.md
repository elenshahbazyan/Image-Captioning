# Image-Captioning

## Project summary

This repository contains a compact, end-to-end image-captioning pipeline developed using the COCO captions dataset. The implementation focuses on clarity, reproducibility and robustness: from raw captions and images to a working encoder–decoder model and recorded training artifacts.

## Project goal

To implement a minimal but complete image-captioning system that demonstrates the full experimental flow: data preprocessing and vocabulary construction, dataset batching and collation, visual inspection of preprocessed samples, a ResNet-based image encoder, an LSTM text decoder with inference sampling, and a stable training procedure with checkpoints and recorded losses.

## What was implemented (concise, factual)

* **Preprocessing and vocabulary**: caption tokenization, frequency counting, vocabulary construction and persistence. Captions are converted to numerical token sequences with explicit start/end markers.

* **Dataset and dataloader**: COCO annotations are read and mapped to image files; the dataset returns image tensors, padded caption tensors and corresponding lengths. Collation handles padding and produces the tuple used by training and inspection routines.

* **Visual inspection utility**: a small script loads a batch, decodes example captions back to text, and displays images with their decoded captions to confirm preprocessing correctness.

* **Encoder (feature extractor)**: a pretrained ResNet-based encoder that outputs a fixed-length embedding per image through a projection layer and normalization. The backbone is frozen by default in the experiments reported here; the implementation supports unfreezing for fine-tuning.

* **Decoder**: an LSTM-based sequence decoder that trains using teacher forcing and supports greedy sampling for inference; it consumes the image embedding and produces word probability sequences over the vocabulary.

* **Training procedure**: a training script that correctly aligns decoder outputs and targets, packs variable-length sequences for efficient loss computation, applies cross-entropy loss with padding ignored, saves per-epoch checkpoints, and records loss history to a CSV and a final plot image.

## Development notes and robustness fixes

During development the codebase required and received the following concrete fixes and improvements, all implemented in the repository:

* Unified vocabulary interface and token index attributes so all modules access the same tokens and indices reliably.
* Added `numericalize`, `save` and `load` functionality for the vocabulary to persist experiments.
* Ensured image transformations (train vs. eval) are available through a single helper with deterministic behavior in evaluation mode.
* Addressed sequence packing alignment: targets are shifted and packed using lengths minus one, and decoder predictions are aligned accordingly to avoid off-by-one mismatches in loss computation.
* Added fallbacks and safe defaults (e.g. single-worker loader on Windows) to avoid platform-specific failures.
* Resolved RNN/dropout warning by matching dropout configuration to the number of LSTM layers.

All fixes were validated by running the inspection utility and by running short training sessions that produced stable losses and saved artifacts.

## Execution and artifacts

All experiments ran from the project root using the COCO dataset placed under `data/coco/`. The runs produced the following artifacts (committed or saved during development):

* A persisted vocabulary file in the `preprocessed/` directory.
* Per-epoch model checkpoints and a best-model checkpoint in the `checkpoints/` directory.
* A training-loss CSV log and a PNG plot summarizing training loss, saved under `preprocessed/`.
* Example visualization outputs produced by the inspection utility (images with decoded captions), used to confirm preprocessing correctness.

The training runs used a frozen encoder by default, and the optimizer was configured to update the decoder and the encoder’s projection/batch-normalization parameters only.

## Files overview (what each file contains, no code excerpts)

* `preprocess.py` — tokenization, vocabulary build/save/load, and image transform utilities.
* `data_loader.py` — COCO dataset implementation, collation and loader helper returning batched tensors and lengths.
* `inspect_batch.py` — utility to visualize decoded captions and verify preprocessing.
* `feature_extractor.py` — ResNet-based encoder with projection to a fixed embedding size.
* `decoder.py` — LSTM decoder with training forward pass and greedy sampling for inference.
* `train.py` — orchestrates training: data loading, forward/backward steps, packing/alignment, checkpointing and logging.
* Directories produced or used by scripts: `preprocessed/` and `checkpoints/`.

## How to reproduce the recorded runs (high level)

1. Place COCO images and caption JSON under the `data/coco/` directory following the structural conventions used in the project.
2. Run the vocabulary builder to create the persisted vocabulary file.
3. Use the inspection utility to verify that images and decoded captions look correct.
4. Run the training script; it accepts command-line options for batch size, number of epochs, and dataset paths. Training saves checkpoints and writes loss logs and a plot at the end.

(Exact command lines and available options are documented as CLI help within each script.)

## Environment and reproducibility details

* Development tested with Python 3.8–3.12 and recent PyTorch/torchvision releases.
* Required Python packages include PyTorch, torchvision, matplotlib, nltk, pillow and tqdm. On systems where `pycocotools` is required, the repository assumes it is installed (Windows users may use the platform-specific wheel where necessary).
* The project uses deterministic evaluation transforms and documented defaults to make runs reproducible given the same data and seed.

## Final statement

The codebase implements and validates a complete image-captioning workflow end-to-end. All implemented components, robustness fixes and artifacts mentioned above are present in the repository. This README documents the exact work performed and the outputs produced during development.

---

If you would like, I can now write this exact text to `README.md` in the project repository. Confirm and I will add the file.
