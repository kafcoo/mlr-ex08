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
python -u training/train.py --seeds 0 1 2 --epochs 15
python training/evaluate.py
python training/export_model.py
```

`prepare_data.py` creates `data/clean` and writes [the data audit](results/data_audit.json). It hashes decoded RGB pixels, excludes every member of groups with conflicting labels, and deduplicates the rest. Validation takes priority over training, so overlapping training copies are excluded. It reserves 20% of the remaining historical training pictures per class for development, using seed 2026. The original validation pictures are a second development source. The historical validation set is reserved for the final test. Original source images are never modified.

It also creates fixed camera-like development and test copies, using seed 314159, moderate crops, rotations and colour changes. These are synthetic robustness checks, not actual camera measurements or additional independent samples.

`train.py` compares four variants, one change at a time:

1. `original_only`: original training data, pretrained MobileNetV3-small fine-tuned end to end.
2. `expanded_only`: add the cleaned historical training photos.
3. `expanded_geo`: also use random crops, flips and rotations.
4. `expanded_color`: also use colour jitter.

All variants use the same architecture, input normalization, batch size 32, cross-entropy loss, Adam at 0.001 and 15 epochs. Each seed resets the pretrained network, replacement head and random generators. Development results select the best checkpoint. Four development balanced accuracies (original/historical × clean/camera-like) receive equal weight, avoiding dominance by the larger source. Cross-entropy breaks ties. The final test is never used by the training loop.

Results are saved after each completed run; rerunning skips completed runs. To change the protocol, put results in a separate experiment directory rather than mixing them with these runs. MPS is used when available; `--device cpu` forces CPU. Fixed seeds control inputs and initialization; exact bitwise MPS reproducibility across hardware/software is not guaranteed.

`evaluate.py` compares mean development scores across seeds and persists the selection **before** running any final-test inference. It chooses the middle seed by development score, avoiding selection of the luckiest individual seed. It also evaluates the previously deployed ONNX model, backed up locally on its first run. All models are compared on the same final test and camera-like test. Checkpoints and the original-model backup stay in the ignored `training/checkpoints` directory.

`export_model.py` exports the selected checkpoint to opset 17, checks the graph, compares ONNX Runtime against PyTorch on the reference image and five random inputs, and updates the root model and expected reference probabilities together. The reference JPEG stays unchanged. The original deployed model and reference values remain recoverable from the repository's initial commits.

## Limits

Exact-pixel duplicate removal does not identify every photograph of the same physical leaf or near-duplicate image. The archive has no reliable plant/session identifiers for a complete group split. Development-set selection and the small original validation set also limit certainty. Confirm improvements with new phone photographs before claiming field accuracy. The live app's uncertainty indicator is a fixed score/margin heuristic, not a trained fifth class or a guarantee of detecting unfamiliar plants.

These are technical measurements generated during development; the course's student-authored discussion should use your own interpretation of the results.
