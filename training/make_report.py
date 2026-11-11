"""Render measured results as a concise technical comparison and static plot."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".local-tools/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("training/results")
NAMES = {"deployed_original": "Original deployed model", "original_only": "Retrained · original data",
         "expanded_only": "+ Historical data", "expanded_geo": "+ Geometry", "expanded_color": "+ Colour"}


def main():
    evaluation = json.loads((RESULTS / "evaluation.json").read_text())
    audit = json.loads((RESULTS / "data_audit.json").read_text())
    exported = json.loads((RESULTS / "export.json").read_text())
    selection, rows = evaluation["selection"], evaluation["models"]
    variants = list(NAMES)
    old = rows[0]
    selected = next(r for r in rows if r["variant"] == selection["selected_variant"] and r["seed"] == selection["selected_seed"])
    table = []
    means, deviations = [], []
    for variant in variants:
        runs = [row for row in rows if row["variant"] == variant]
        clean = [r["metrics"]["test"]["accuracy"] * 100 for r in runs]
        camera = [r["metrics"]["camera_test"]["accuracy"] * 100 for r in runs]
        dev = [r["dev_score"] * 100 for r in runs]
        table.append(f"| {NAMES[variant]} | {len(runs)} | {np.mean(dev):.2f} | {np.mean(clean):.2f} ± {np.std(clean):.2f} | {np.mean(camera):.2f} ± {np.std(camera):.2f} |")
        means.append([np.mean(clean), np.mean(camera)])
        deviations.append([np.std(clean), np.std(camera)])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    x = np.arange(len(variants))
    for i, (label, colour) in enumerate([("Clean test", "#285a42"), ("Camera-like test", "#b89b52")]):
        axes[0].bar(x + (i - .5) * .36, np.array(means)[:, i], .36,
                    yerr=np.array(deviations)[:, i], capsize=3, label=label, color=colour)
    axes[0].set_xticks(x, ["Original\ndeployed", "Original\nretrained", "+ More\ndata", "+ Geometry", "+ Colour"])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Same held-out test · 120 photos")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", alpha=.2)
    axes[0].set_axisbelow(True)
    classes = audit["classes"]
    x = np.arange(len(classes))
    for i, (row, label, colour) in enumerate([(old, "Original deployed", "#b8c1b1"), (selected, "Selected checkpoint", "#285a42")]):
        axes[1].bar(x + (i - .5) * .36, np.array(row["metrics"]["test"]["per_class_recall"]) * 100,
                    .36, color=colour, label=label)
    axes[1].set_xticks(x, classes)
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Clean test recall · 30 photos per class")
    axes[1].set_ylabel("Recall (%)")
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="y", alpha=.2)
    axes[1].set_axisbelow(True)
    fig.suptitle("Leaf lens · controlled data and augmentation comparison", fontsize=15)
    fig.savefig(RESULTS / "comparison.png", dpi=170)
    plt.close(fig)

    def percent(row, name):
        return f"{row['metrics'][name]['accuracy'] * 100:.2f}%"

    confusion = selected["metrics"]["test"]["confusion"]
    matrix = "\n".join("| " + label + " | " + " | ".join(map(str, counts)) + " |" for label, counts in zip(classes, confusion))
    reference = "\n".join(f"| {name} | {prob:.10f} |" for name, prob in exported["reference_probabilities"].items())
    text = f"""# Leaf lens: measured training comparison

Archived first-study technical measurements, 11 November 2026. These 12 runs describe the previous checkpoint and its original JPEG behavior. See [NETWORK_RESULTS.md](NETWORK_RESULTS.md) and [release_info.json](release_info.json) for the current model and resolved JPEG pipeline. Historical test scores below must not be attributed to the new release.

## Result

The checkpoint selected **using development data before any final test inference** is `{selection['selected_variant']}`, seed **{selection['selected_seed']}**. The representative seed is the middle run by development score; the variant is selected by its mean development score over three seeds.

On the same held-out historical test, the original deployed model scored **{percent(old, 'test')}** and the selected checkpoint scored **{percent(selected, 'test')}**. On fixed camera-like versions of those same photos, they scored **{percent(old, 'camera_test')}** and **{percent(selected, 'camera_test')}**. The two test views are correlated, not 240 independent photographs.

![Accuracy comparison and per-class recall](training/results/comparison.png)

## Controlled comparison

All new runs use ImageNet-pretrained MobileNetV3-small, 96 × 96 input, identical normalization, batch size 32, Adam at 0.001 and 15 epochs. The entire network is fine-tuned. Each run keeps its best development checkpoint, which differs from the original notebook's final-epoch export. Changes are cumulative: more data, then geometric augmentation, then colour jitter. Each new variant uses seeds 0, 1 and 2.

| Model / cumulative change | Runs | Dev score (%) | Clean test accuracy (%) | Camera-like test accuracy (%) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

The dominant gain is from adding historical training data. Geometric augmentation gives a smaller mean clean-test improvement; colour jitter does **not** improve mean clean-test accuracy over geometry in these runs. Colour's mean development advantage over geometry is only 0.23 percentage points, smaller than its between-seed spread. These experiments do not establish that colour jitter is the better recipe. The deployed checkpoint follows the development-only selection rule fixed before test evaluation; it was not switched to the best-looking test run afterwards. Further recipe changes need a new independent test set.

`±` is the population standard deviation **across seeds**, not a confidence interval over photos. The original deployed model is a single fixed checkpoint; its displayed zero spread is not an uncertainty estimate. Clean test has 30 images per class, so ordinary and balanced accuracy coincide there. One error changes overall accuracy by 0.83 percentage points. Small seed differences do not establish statistical significance or field generalization.

Development score gives equal weight to balanced accuracy on original validation, historical development, and a fixed camera-like view of each. No test result was used for checkpoint, variant or seed selection. See [selection.json](training/results/selection.json) and [all per-run metrics](training/results/evaluation.json).

## Data cleaning and split

- Inputs: **956** photos (136 original, 820 historical).
- Removed **{audit['excluded_count']}** files: **{audit['exclusion_reasons']['duplicate pixels']}** redundant decoded-pixel duplicates and **{audit['exclusion_reasons']['conflicting labels']}** photos in conflicting-label groups. Conflicting classes were apple/cherry; both copies were excluded instead of guessing the label.
- For duplicates crossing train/validation, the validation copy was preserved and the training copy was removed.
- Original-only training: **107** photos. Expanded training: **649** photos (166 apple, 166 cherry, 155 chestnut, 162 maple).
- Development: **29** original validation photos plus **137** reserved historical training photos (stratified split seed 2026). All variants use these same development sets.
- Final test: **120** historical validation photos. No exact decoded-pixel overlap with any development/training set.
- Camera-like transforms were generated once with seed 314159: crop scale 0.7–1.0, rotation ±20°, brightness/contrast ±0.25, saturation ±0.2 and hue ±0.03. Geometry training also includes horizontal and vertical flips.

The [full audit](training/results/data_audit.json) lists every retained image, destination and exclusion. Raw data and checkpoints remain local. Exact duplicates were removed; unrecognized near-duplicates or photos of the same physical leaf may remain because plant/session identifiers are unavailable.

## Selected model: clean-test confusion matrix

Rows are true classes; columns are predictions.

| True / predicted | apple | cherry | chestnut | maple |
| --- | ---: | ---: | ---: | ---: |
{matrix}

## Export and web checks

- ONNX opset **{exported['opset']}**, input **1 × 3 × 96 × 96**, four output logits, **{exported['model_bytes']:,} bytes**.
- Class names and preprocessing are also embedded in ONNX under `model_info`, as required by Task 2e and the course camera app.
- Model SHA-256: `{exported['model_sha256']}`.
- Maximum PyTorch/ONNX logit difference on five fixed random inputs: **{exported['max_logit_error']:.3g}**. The reference-image probabilities were checked separately.
- The reference JPEG is unchanged; [check_expected.json](check_expected.json) now corresponds to the newly trained model. Previous values remain in Git history.

| Reference class | PyTorch expected probability |
| --- | ---: |
{reference}

Browser verification results are documented in [the README](README.md) and [browser measurements](training/results/browser_validation.json). Native JPEG decoding differs between WebKit and Pillow/Chrome: WebKit's maximum probability discrepancy is about 0.00118 (0.118 percentage points). With the [lossless fixture containing the exact Pillow RGB pixels](training/results/reference_pixels.png), both browsers match PyTorch within 0.00000025. The JPEG-aware reference check therefore uses absolute tolerance 0.002; the separate same-pixels model check uses 0.00001. This distinguishes decoder variation from an export/preprocessing failure.

Passing the JPEG-aware tolerance does **not** mean agreeing to three decimals. WebKit currently rounds apple to 0.811 while PyTorch rounds it to 0.810. The site reports these two checks separately. See [the requirement audit and remaining student tasks](AUDIT.md); native-JPEG three-decimal agreement remains an open requirement on WebKit.

The browser's low-confidence display uses a fixed heuristic: top probability below 0.65 or top-two margin below 0.15. It preserves all four class probabilities. It is not a learned unknown class, and high confidence does not prove that a leaf belongs to a known species. A fifth-class detector needs representative negative training and evaluation data.

## Reproduce and interpret

See [training instructions](training/README.md), [pinned dependencies](training/requirements.txt), [data preparation](training/prepare_data.py), [training loop](training/train.py), [evaluation](training/evaluate.py), and [ONNX export](training/export_model.py). Fixed seeds do not guarantee bitwise-identical MPS results on different hardware/software. The original deployed model is recoverable from the initial Git commits; its recorded SHA-256 is `{selection['baseline_sha256']}`.

Synthetic camera-like accuracy estimates robustness to those particular transforms. New real phone photographs, ideally grouped by plant and capture session, are needed for an independent field test. No physical iPhone camera was available during this run. Do not interpret the 29-photo original development score as an independent test.
"""
    Path("TRAINING_RESULTS.md").write_text(text)
    print("Wrote TRAINING_RESULTS.md and training/results/comparison.png")


if __name__ == "__main__":
    main()
