"""Select using development data first; only then evaluate every model on test."""
import hashlib
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from train import MEAN, STD, EVAL_SETS, ROOT, RESULTS, CHECKPOINTS, evaluate
from networks import build_model


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="training")
    parser.add_argument("--baseline", default="training/checkpoints/original.onnx")
    parser.add_argument("--baseline-sha256", default="ae31c68aa9920808565fed2eaf50164647f90887b396a9fafe02fb9e4f034f7f")
    parser.add_argument("--dev-only", action="store_true", help="Do not open the already-examined historical test")
    args = parser.parse_args()
    results, checkpoints = Path(args.run_dir) / "results", Path(args.run_dir) / "checkpoints"
    baseline = Path(args.baseline)
    if not baseline.exists():
        parser.error("Original baseline is missing. Supply --baseline explicitly; never substitute the current deployed model. See training/README.md.")
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != args.baseline_sha256:
        parser.error("Baseline SHA-256 does not match the fixed comparison model")
    runs = [json.loads(path.read_text()) for path in sorted(results.glob("*_seed*.json"))]
    if not runs:
        parser.error("No completed runs in the selected --run-dir")
    for run in runs:
        path = checkpoints / f"{run['variant']}_seed{run['seed']}.pt"
        if not path.exists():
            parser.error(f"Missing checkpoint {path}; published JSON metrics alone cannot be resumed")
    torch.set_num_threads(4)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    evaluation_sets = EVAL_SETS if args.dev_only else EVAL_SETS + ["test", "camera_test"]
    loaders = {name: DataLoader(datasets.ImageFolder(ROOT / name, transform), batch_size=64)
               for name in evaluation_sets}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    old = ort.InferenceSession(str(baseline), sess_options=options, providers=["CPUExecutionProvider"])
    old_dev = {name: onnx_metrics(old, loaders[name]) for name in EVAL_SETS}
    old_score = float(np.mean([m["balanced_accuracy"] for m in old_dev.values()]))
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
                     selected_architecture=representative.get("architecture", "mobilenet_v3_small"),
                     evaluation_scope="development only; new phone test pending" if args.dev_only else "archived historical test comparison",
                     deploy_candidate=representative["best_dev_score"] > old_score,
                     baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())
    # Persist the choice BEFORE opening the test loader for any model.
    (results / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("Selection fixed before test:", json.dumps(selection), flush=True)

    rows = [dict(variant="deployed_original", seed=None, dev_score=old_score,
                 metrics={**old_dev, **{name: onnx_metrics(old, loaders[name]) for name in evaluation_sets if name not in EVAL_SETS}})]
    for run in runs:
        model = build_model(run.get("architecture", "mobilenet_v3_small"))
        model.load_state_dict(torch.load(checkpoints / f"{run['variant']}_seed{run['seed']}.pt", weights_only=True, map_location="cpu"))
        row = dict(variant=run["variant"], seed=run["seed"], best_epoch=run["best_epoch"],
                   dev_score=run["best_dev_score"], metrics={name: evaluate(model, loader, "cpu") for name, loader in loaders.items()})
        rows.append(row)
        print(f"{run['variant']} seed={run['seed']} evaluated={list(row['metrics'])}", flush=True)
    (results / "evaluation.json").write_text(json.dumps(dict(selection=selection, models=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
