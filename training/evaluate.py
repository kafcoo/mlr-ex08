"""Select using development data first; only then evaluate every model on test."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from train import MEAN, STD, EVAL_SETS, ROOT, RESULTS, CHECKPOINTS, evaluate


def onnx_metrics(session, loader):
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    loss_sum = 0.
    with torch.inference_mode():
        for images, labels in loader:
            # The supplied baseline has a fixed batch dimension of one.
            logits = torch.from_numpy(np.concatenate([
                session.run(None, {session.get_inputs()[0].name: image[None].numpy()})[0]
                for image in images]))
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            for target, prediction in zip(labels, logits.argmax(1)):
                confusion[target, prediction] += 1
    recall = confusion.diag() / confusion.sum(1).clamp(min=1)
    return dict(accuracy=(confusion.diag().sum() / confusion.sum()).item(),
                balanced_accuracy=recall.mean().item(), per_class_recall=recall.tolist(),
                cross_entropy=loss_sum / len(loader.dataset), confusion=confusion.tolist(), count=len(loader.dataset))


def main():
    torch.set_num_threads(4)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    loaders = {name: DataLoader(datasets.ImageFolder(ROOT / name, transform), batch_size=64)
               for name in EVAL_SETS + ["test", "camera_test"]}
    baseline = CHECKPOINTS / "original.onnx"
    if not baseline.exists():
        shutil.copy2("leaf_model.onnx", baseline)
        shutil.copy2("check_expected.json", CHECKPOINTS / "original_check_expected.json")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    old = ort.InferenceSession(str(baseline), sess_options=options, providers=["CPUExecutionProvider"])
    old_dev = {name: onnx_metrics(old, loaders[name]) for name in EVAL_SETS}
    old_score = float(np.mean([m["balanced_accuracy"] for m in old_dev.values()]))
    runs = [json.loads(path.read_text()) for path in sorted(RESULTS.glob("*_seed*.json"))]
    summaries = []
    for variant in sorted({run["variant"] for run in runs}):
        scores = [run["best_dev_score"] for run in runs if run["variant"] == variant]
        summaries.append(dict(variant=variant, mean_dev_score=float(np.mean(scores)),
                              std_dev_score=float(np.std(scores)), seeds=len(scores)))
    chosen = max(summaries, key=lambda row: row["mean_dev_score"])
    candidates = sorted([run for run in runs if run["variant"] == chosen["variant"]],
                        key=lambda run: (run["best_dev_score"], run["seed"]))
    # Deploy the middle seed by development score rather than the luckiest run.
    representative = candidates[len(candidates) // 2]
    selection = dict(criterion="mean of four development balanced accuracies; variant mean over seeds; median seed",
                     baseline_dev_score=old_score, variant_summaries=summaries,
                     selected_variant=chosen["variant"], selected_seed=representative["seed"],
                     selected_checkpoint=f"{chosen['variant']}_seed{representative['seed']}.pt",
                     deploy_candidate=representative["best_dev_score"] > old_score,
                     baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())
    # Persist the choice BEFORE opening the test loader for any model.
    (RESULTS / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("Selection fixed before test:", json.dumps(selection), flush=True)

    rows = [dict(variant="deployed_original", seed=None, dev_score=old_score,
                 metrics={**old_dev, **{name: onnx_metrics(old, loaders[name]) for name in ["test", "camera_test"]}})]
    for run in runs:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
        model.load_state_dict(torch.load(CHECKPOINTS / f"{run['variant']}_seed{run['seed']}.pt", weights_only=True, map_location="cpu"))
        row = dict(variant=run["variant"], seed=run["seed"], best_epoch=run["best_epoch"],
                   dev_score=run["best_dev_score"], metrics={name: evaluate(model, loader, "cpu") for name, loader in loaders.items()})
        rows.append(row)
        print(f"{run['variant']} seed={run['seed']} test={row['metrics']['test']['accuracy']:.4f} camera={row['metrics']['camera_test']['accuracy']:.4f}", flush=True)
    (RESULTS / "evaluation.json").write_text(json.dumps(dict(selection=selection, models=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
