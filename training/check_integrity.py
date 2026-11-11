"""Check the archived study, current export and source data without retraining."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image


def main():
    results = Path("training/results")
    checks = []
    audit = json.loads((results / "data_audit.json").read_text())
    classes = audit["classes"]
    destinations = {}
    for image in audit["manifest"]:
        with Image.open(image["path"]) as source:
            rgb = source.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        assert digest == image["pixel_sha256"], f"Source changed: {image['path']}"
        for name in image["destinations"]:
            destinations.setdefault(name, set()).add(digest)
    for name in ["original_val", "historical_dev", "test"]:
        for other in ["original_train", "expanded_train"]:
            assert not destinations[name] & destinations[other], f"Leakage: {name}/{other}"
    for name, other in [("original_val", "historical_dev"), ("original_val", "test"), ("historical_dev", "test")]:
        assert not destinations[name] & destinations[other], f"Leakage: {name}/{other}"
    for folder, counts in audit["counts"].items():
        for label, count in counts.items():
            assert len(list((Path("data/clean") / folder / label).glob("*"))) == count
    checks.append("935 source RGB hashes, recorded counts and train/dev/test separation")

    runs = [json.loads(p.read_text()) for p in sorted(results.glob("*_seed*.json"))]
    assert len(runs) == 12
    for run in runs:
        assert len(run["history"]) == run["epochs"] == 15
        assert [row["epoch"] for row in run["history"]] == list(range(1, 16))
        assert abs(run["best_dev_score"] - max(row["dev_score"] for row in run["history"])) < 1e-8
        best = run["history"][run["best_epoch"] - 1]
        assert best["dev_score"] == run["best_dev_score"]
        for row in run["history"]:
            assert np.isfinite(row["train_loss"]) and row["train_loss"] >= 0
            assert abs(row["dev_score"] - np.mean([m["balanced_accuracy"] for m in row["metrics"].values()])) < 1e-8
    checks.append("12 runs × 15 epochs: finite losses, correct development aggregates and best epochs")

    evaluation = json.loads((results / "evaluation.json").read_text())
    assert len(evaluation["models"]) == 13
    for model in evaluation["models"]:
        for name, metric in model["metrics"].items():
            matrix = np.array(metric["confusion"])
            assert matrix.shape == (4, 4) and np.all(matrix >= 0)
            counts = np.array([audit["counts"][name][label] for label in classes])
            np.testing.assert_array_equal(matrix.sum(1), counts)
            assert matrix.sum() == metric["count"]
            assert abs(np.trace(matrix) / matrix.sum() - metric["accuracy"]) < 1e-7
            recalls = matrix.diagonal() / matrix.sum(1)
            np.testing.assert_allclose(recalls, metric["per_class_recall"], atol=1e-7)
            assert abs(recalls.mean() - metric["balanced_accuracy"]) < 1e-7
    checks.append("13 models × 6 datasets: confusion matrices, sample counts, accuracy and recall")

    selection = json.loads((results / "selection.json").read_text())
    scores = {variant: np.mean([r["best_dev_score"] for r in runs if r["variant"] == variant])
              for variant in {r["variant"] for r in runs}}
    assert max(scores, key=scores.get) == selection["selected_variant"]
    candidates = sorted([r for r in runs if r["variant"] == selection["selected_variant"]],
                        key=lambda r: (r["best_dev_score"], r["seed"]))
    assert candidates[1]["seed"] == selection["selected_seed"]
    assert selection == evaluation["selection"]
    checks.append("Archived variant/median-seed selection agrees with development-only rule")

    model_path = Path("leaf_model.onnx")
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    info = json.loads(Path("model_info.json").read_text())
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    assert json.loads(metadata["model_info"]) == info
    release_path = Path("release_info.json")
    release_run = Path(json.loads(release_path.read_text())["run_dir"]) if release_path.exists() else Path("training")
    export = json.loads((release_run / "results/export.json").read_text())
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == export["model_sha256"]
    assert model_path.stat().st_size == export["model_bytes"]
    image = np.asarray(Image.open("check_image.jpg").convert("RGB"), dtype=np.float32) / 255
    pixels = np.asarray(Image.open(results / "reference_pixels.png").convert("RGB"))
    np.testing.assert_array_equal(pixels, np.asarray(Image.open("check_image.jpg").convert("RGB")))
    x = ((image - np.array(info["mean"])) / np.array(info["std"])).transpose(2, 0, 1)[None].astype(np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, 96, 96]
    logits = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    expected = json.loads(Path("check_expected.json").read_text())
    np.testing.assert_allclose(probabilities, [expected[label] for label in classes], rtol=1e-4, atol=1e-6)
    checks.append("Current ONNX graph, embedded/JSON metadata, SHA-256, NCHW and reference probabilities")
    with zipfile.ZipFile("ex08_for_agent.zip") as archive:
        for name in ["leaf_model.onnx", "model_info.json", "check_expected.json", "check_image.jpg"]:
            assert archive.read(name) == Path(name).read_bytes(), f"Stale ZIP entry: {name}"
    checks.append("Export ZIP contains the exact current model and all three companion files")
    output = dict(status="PASS", checks=checks, model_sha256=export["model_sha256"])
    (results / "integrity_checks.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
