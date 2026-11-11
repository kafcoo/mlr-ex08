# Reproducing the leaf experiments

Use Python 3.12 and run commands from the repository root. These scripts retain the notebook's readable PyTorch loop. `networks.py` contains the Task 3 LeafNet/LeafNetBN, a small GAP option, and the existing MobileNet baseline. There is no new training framework.

## Environment and data

```sh
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r training/requirements.txt
```

For the complete notebook as well, install `training/notebook-requirements.txt` and select that environment as its kernel. The source notebook stays local; its Task 3 definitions are the source for `check_networks.py` parity tests.

Keep the original 107 train / 29 validation images in `data/leaves/{train,val}/{apple,cherry,chestnut,maple}`. Extract [the historical archive](https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/ex08_leaves_images.zip) into `data/historical`, producing `data/historical/leaves/{train,val}`. Original photos and checkpoints are not published.

`prepare_data.py` hashes decoded RGB pixels, removes duplicates, excludes conflicting labels, keeps validation copies over overlapping train copies and reserves a fixed 20% historical dev split (seed 2026). Camera-like copies are generated once (seed 314159). The resulting counts are 649 train, 29 original dev, 137 historical dev, and 120 historical test. This existing `data/clean` is unchanged in the new study. Before adding other source data, isolate its prepared folders and audit rather than replacing the archived study's manifest.

## New 21-run study

```sh
python training/train.py --run-dir training/runs/networks_reproduce --variants mobile_geo mobile_frozen mobile_cosine mobile_smoothing leaf_geo leaf_bn_geo leaf_gap_geo --seeds 0 1 2 --epochs 15
mkdir -p training/checkpoints
curl -fL https://raw.githubusercontent.com/kafcoo/mlr-ex08/0303611e72702491b5e38114620f30c72a9ea7c5/leaf_model.onnx -o training/checkpoints/before_networks.onnx
python training/evaluate.py --run-dir training/runs/networks_reproduce --baseline training/checkpoints/before_networks.onnx --baseline-sha256 22bde880732116f01725c7b7d9bd58f14d7a5be62be593d3ad71657ebd9ad64a --dev-only
python training/export_model.py --run-dir training/runs/networks_reproduce --output-dir exports/candidate
```

Use a new run directory for reproduction. The archived result JSONs are not weights. A run resumes only if its checkpoint and matching protocol exist; otherwise missing weights trigger training. Changed data/settings are rejected in the same directory. Protocol fingerprints cover actual train/dev images, versions, transforms and architecture source. Fixed seeds control initialization and sampling; MPS is not guaranteed bitwise-identical across hardware/software.

All seven variants use geometric augmentation, Adam 0.001, batch 32 and 15 epochs. `mobile_frozen` trains only 4,100 final-layer parameters, freezing BN statistics and dropout too. `mobile_cosine` changes only the schedule; `mobile_smoothing` changes only training-label smoothing to 0.05. `leaf_bn_geo` adds BN to Task 3 LeafNet; `leaf_gap_geo` replaces flattening with a spatial mean and adjusts fc input size. MobileNet is ImageNet-pretrained; the custom networks start from scratch, so this comparison does not isolate architecture from pretraining.

Best checkpoints use the mean of four development balanced accuracies (original/historical × clean/camera-like). Variant selection uses the mean over three seeds; the representative seed is the middle dev-scoring seed. `--dev-only` never builds or reads historical test loaders. Historical test results were already examined in the first study, so new confirmatory claims need a new phone/session test.

## Original four-recipe study

The original 12 runs remain in `training/results`; do not overwrite them. Reproduce elsewhere with default variants:

```sh
python training/train.py --run-dir training/runs/original_reproduce --seeds 0 1 2 --epochs 15
curl -fL https://raw.githubusercontent.com/kafcoo/mlr-ex08/1029091ee246182edfd05f3d7c006b1812e2d84e/leaf_model.onnx -o training/checkpoints/original.onnx
python training/evaluate.py --run-dir training/runs/original_reproduce --baseline training/checkpoints/original.onnx
```

The fixed original SHA-256 is `ae31c68aa9920808565fed2eaf50164647f90887b396a9fafe02fb9e4f034f7f`. The evaluator refuses a missing or wrong baseline rather than silently copying the current released model. That study compares original data, added historical train data, geometry, then ColorJitter. Its historical test results are archival; repeating inference does not create a fresh independent test.

## Export, tests and release

`export_model.py` writes model, metadata, reference probabilities and a four-file ZIP into `--output-dir` (default root). It embeds `model_info` inside ONNX, checks the graph and compares ONNX Runtime against PyTorch on the unchanged JPEG plus five fixed random inputs. The released model's study/seed/SHA is in root `release_info.json`; update this provenance and the HTML's model revision together before publishing another model. Use an isolated candidate folder and verify first.

```sh
python training/check_integrity.py
python training/test_reproducibility.py
python training/check_networks.py
python training/check_new_study.py
```

The integrity checks need local original/clean data and the current root export ZIP. Network parity checks also need the local corrected notebook. `profile_networks.py` exports representative candidates for local browser parity/timing tests; all checkpoint files remain ignored. Measured WASM times are desktop Mac timings, not iPhone timings.

## Notebook, JPEG and student work

The user-authorized corrected notebook uses the augmented loader, seeded shuffle and filtered image extensions. It passed 30/30 cells and 15/15 semantic checks. Its original version is backed up locally. Task 2 exports into `exports/task2/`; the original archived notebook would overwrite root exports, so audits continue to run in an isolated working directory.

The page uses one pinned MozJPEG/WASM decoder for JPEG uploads, including EXIF rotation/mirroring. The unchanged reference passes strict 1e-5 probability tolerance and three-decimal agreement in both WebKit and Chrome; all eight EXIF orientations match Pillow pixels. Camera frames are already decoded. The extra library/WASM downloads from the CDN, but photos remain on the device.

See [NETWORK_RESULTS.md](../NETWORK_RESULTS.md), [REPORT.md](../REPORT.md) and [REVIEW.md](../REVIEW.md). The report and review include the phone trial and reference check, with links to the screenshots. Final changes must be pushed within 48 hours after the lab. Internal audit notes and notebook backups are local files, not required deliverables.
