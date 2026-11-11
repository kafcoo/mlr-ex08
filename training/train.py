"""The notebook's MobileNet training loop, with controlled data/augmentation variants."""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN = [.485, .456, .406]
STD = [.229, .224, .225]
EVAL_SETS = ["original_val", "historical_dev", "camera_original_val", "camera_historical_dev"]
ROOT = Path("data/clean")
RESULTS = Path("training/results")
CHECKPOINTS = Path("training/checkpoints")


def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    loss_sum = 0.
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device)).cpu()
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            for target, prediction in zip(labels, logits.argmax(1)):
                confusion[target, prediction] += 1
    recall = confusion.diag() / confusion.sum(1).clamp(min=1)
    return dict(accuracy=(confusion.diag().sum() / confusion.sum()).item(),
                balanced_accuracy=recall.mean().item(), per_class_recall=recall.tolist(),
                cross_entropy=loss_sum / len(loader.dataset), confusion=confusion.tolist(),
                count=len(loader.dataset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["original_only", "expanded_only", "expanded_geo", "expanded_color"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    normalise = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    evaluation = transforms.Compose(normalise)
    loaders = {name: DataLoader(datasets.ImageFolder(ROOT / name, evaluation), batch_size=64, num_workers=0)
               for name in EVAL_SETS}

    for seed in args.seeds:
        for variant in args.variants:
            result_path = RESULTS / f"{variant}_seed{seed}.json"
            if result_path.exists():
                print("Already completed:", result_path, flush=True)
                continue
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "mps":
                torch.mps.manual_seed(seed)
            augment = []
            if variant in ["expanded_geo", "expanded_color"]:
                augment = [transforms.RandomResizedCrop(96, scale=(.7, 1.0)),
                           transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                           transforms.RandomRotation(20)]
            if variant == "expanded_color":
                augment += [transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.2, hue=.03)]
            train_name = "original_train" if variant == "original_only" else "expanded_train"
            dataset = datasets.ImageFolder(ROOT / train_name, transforms.Compose(augment + normalise))
            loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(seed))
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            history = []
            best_score, best_loss = -1., float("inf")
            start = time.perf_counter()
            for epoch in range(args.epochs):
                model.train()
                loss_sum = 0.
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * len(labels)
                metrics = {name: evaluate(model, val_loader, device) for name, val_loader in loaders.items()}
                # Equal weight to both data sources and to clean/camera-like views.
                score = float(np.mean([m["balanced_accuracy"] for m in metrics.values()]))
                val_loss = float(np.mean([m["cross_entropy"] for m in metrics.values()]))
                history.append(dict(epoch=epoch + 1, train_loss=loss_sum / len(dataset), dev_score=score, metrics=metrics))
                if (score, -val_loss) > (best_score, -best_loss):
                    best_score, best_loss, best_epoch = score, val_loss, epoch + 1
                    best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                print(f"{variant} seed={seed} epoch={epoch+1:02d} loss={loss_sum/len(dataset):.4f} dev={score:.4f} elapsed={time.perf_counter()-start:.1f}s", flush=True)
            torch.save(best_state, CHECKPOINTS / f"{variant}_seed{seed}.pt")
            result = dict(variant=variant, seed=seed, epochs=args.epochs, best_epoch=best_epoch,
                          best_dev_score=best_score, best_dev_loss=best_loss, train_count=len(dataset),
                          classes=dataset.classes, seconds=time.perf_counter() - start, device=str(device),
                          torch_version=torch.__version__, torchvision_version=__import__("torchvision").__version__,
                          history=history)
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            del model, optimizer
            if device.type == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
