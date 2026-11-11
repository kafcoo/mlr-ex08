# Reproducing the leaf experiments

These scripts retain the notebook's readable MobileNetV3-small training loop and 96 × 96 RGB inputs. No training framework is added. Run commands from the repository root, using Python 3.12.

## Data

Keep the original 107 training and 29 validation photos in `data/leaves/{train,val}/{apple,cherry,chestnut,maple}`. They are local inputs and are not published with this repository.

Download [the historical ZIP](https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/ex08_leaves_images.zip) and extract it inside `data/historical`, producing `data/historical/leaves/train` and `data/historical/leaves/val`. The [exercise](https://tomasvicar.github.io/MLR-public/exercises/ex08.html) describes it as photographs from other students and phones in previous years. Its `source_files.csv` preserves provenance.

## Commands

```sh
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r training/requirements.txt
python training/prepare_data.py
python -u training/train.py --run-dir training/runs/reproduce --seeds 0 1 2 --epochs 15
# Recover the fixed original Task 2 baseline; verify its SHA-256 during evaluation.
mkdir -p training/checkpoints
curl -fL https://raw.githubusercontent.com/kafcoo/mlr-ex08/1029091ee246182edfd05f3d7c006b1812e2d84e/leaf_model.onnx -o training/checkpoints/original.onnx
python training/evaluate.py --run-dir training/runs/reproduce
python training/export_model.py --run-dir training/runs/reproduce --output-dir .local-tools/reproduced-export
```

`prepare_data.py` creates `data/clean` and writes [the data audit](results/data_audit.json). It hashes decoded RGB pixels, excludes every member of groups with conflicting labels, and deduplicates the rest. Validation takes priority over training, so overlapping training copies are excluded. It reserves 20% of the remaining historical training pictures per class for development, using seed 2026. The original validation pictures are a second development source. The historical validation set is reserved for the final test. Original source images are never modified.

It also creates fixed camera-like development and test copies, using seed 314159, moderate crops, rotations and colour changes. These are synthetic robustness checks, not actual camera measurements or additional independent samples.

`train.py` compares four variants, one change at a time:

1. `original_only`: original training data, pretrained MobileNetV3-small fine-tuned end to end.
2. `expanded_only`: add the cleaned historical training photos.
3. `expanded_geo`: also use random crops, flips and rotations.
4. `expanded_color`: also use colour jitter.

All variants use the same architecture, input normalization, batch size 32, cross-entropy loss, Adam at 0.001 and 15 epochs. Each seed resets the pretrained network, replacement head and random generators. Development results select the best checkpoint. Four development balanced accuracies (original/historical × clean/camera-like) receive equal weight, avoiding dominance by the larger source. Cross-entropy breaks ties. The final test is never used by the training loop.

Results are saved after each completed run. A run is skipped only if its matching checkpoint and recorded protocol exist. `protocol.json` fingerprints the actual training/development image files, transforms, versions, device and optimizer settings; test files are not read to compute it. Missing weights cause retraining. Changed settings/data are rejected in the same run directory. The original `training/results` records are archived: use a separate `--run-dir` for another experiment. MPS is used when available; `--device cpu` forces CPU. Fixed seeds control inputs and initialization; exact bitwise MPS reproducibility across hardware/software is not guaranteed.

`evaluate.py` compares mean development scores across seeds and persists the selection **before** running any final-test inference. It chooses the middle seed by development score. It requires an explicit existing baseline with matching SHA-256 and checks that every required checkpoint exists before test inference. It never substitutes the current deployed model for a missing original baseline. The fixed original SHA-256 is `ae31c68aa9920808565fed2eaf50164647f90887b396a9fafe02fb9e4f034f7f`. For a consciously changed future baseline, pass both `--baseline` and `--baseline-sha256` and label the comparison accordingly.

`export_model.py` exports the selected checkpoint to opset 17, embeds `model_info` in ONNX as in Task 2e, checks the graph and compares ONNX Runtime against PyTorch on the reference image and five random inputs. It writes all four companion files and `ex08_for_agent.zip` into `--output-dir`. The default output is the repository root, so use a separate output directory for experiments. The reference JPEG stays unchanged. The original model remains recoverable from the initial Git commits.

## Verification and student-owned work

```sh
# Requires the local original/clean data and current root export bundle.
python training/check_integrity.py
# Short CPU smoke test in a temporary directory; does not retrain the archived study.
python training/test_reproducibility.py
```

[AUDIT.md](../AUDIT.md) records requirement coverage, the unchanged notebook's clean execution, open defects and the proposed next experiment matrix. The original notebook has a Task 3d bug: its loop consumes `my_train_loader`, although `aug_train_loader` was created. This must be corrected and rerun by the student; the historical Task 4 MobileNet runs are independent of that notebook loop.

The browser now distinguishes **JPEG tolerance PASS** from **three-decimal agreement**. WebKit passes the former but currently differs on the latter; the lossless same-RGB fixture confirms the numerical model/preprocessing path. Do not claim this fulfils the native-JPEG three-decimal check without recording the discrepancy in your own review.

The student must write `REVIEW.md` and `REPORT.md`, attach a real phone screenshot, and try the export in the course camera app. Agent-generated technical records do not replace those files. Execute the notebook in a separate working directory: its original Task 2 export cells overwrite the root model/reference files if run from the repository root.

## Limits

Exact-pixel duplicate removal does not identify every photograph of the same physical leaf or near-duplicate image. The archive has no reliable plant/session identifiers for a complete group split. The historical 120-image test has now been examined; future model changes need a new independent phone/session test for confirmatory results. Development-set selection and the small original validation set also limit certainty. The app's uncertainty indicator is a fixed heuristic, not a trained fifth class or a guarantee of detecting unfamiliar plants.

These are technical measurements generated during development; the course's student-authored discussion should use your own interpretation of the results.
