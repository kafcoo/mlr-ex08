"""Clean exact decoded-pixel duplicates and make fixed train/dev/test folders.

Run from the repository root. Original photos are never changed.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil

from PIL import Image
import torch
from torchvision import transforms

CLASSES = ["apple", "cherry", "chestnut", "maple"]
OUT = Path("data/clean")
RESULTS = Path("training/results")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for source, root in [("original", Path("data/leaves")),
                         ("historical", Path("data/historical/leaves"))]:
        for split in ["train", "val"]:
            for label in CLASSES:
                for path in sorted((root / split / label).glob("*.jpg")):
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                    groups[digest].append(dict(path=str(path), source=source, split=split, label=label))

    kept, excluded = [], []
    for digest, images in sorted(groups.items()):
        if len({x["label"] for x in images}) > 1:
            excluded.extend(dict(**x, reason="conflicting labels", pixel_sha256=digest) for x in images)
            continue
        # Keep validation over training, and original photos over historical ones.
        images.sort(key=lambda x: (x["split"] != "val", x["source"] != "original", x["path"]))
        kept.append(dict(**images[0], pixel_sha256=digest))
        excluded.extend(dict(**x, reason="duplicate pixels", pixel_sha256=digest) for x in images[1:])

    historical_train = [x for x in kept if x["source"] == "historical" and x["split"] == "train"]
    dev_paths = set()
    rng = random.Random(2026)
    for label in CLASSES:
        images = sorted(x["path"] for x in historical_train if x["label"] == label)
        rng.shuffle(images)
        dev_paths.update(images[:round(len(images) * .2)])

    if OUT.exists():
        shutil.rmtree(OUT)
    names = ["original_train", "expanded_train", "original_val", "historical_dev", "test",
             "camera_original_val", "camera_historical_dev", "camera_test"]
    for name in names:
        for label in CLASSES:
            (OUT / name / label).mkdir(parents=True, exist_ok=True)

    manifest = []
    for image in kept:
        source, split = image["source"], image["split"]
        if source == "original":
            targets = ["original_train", "expanded_train"] if split == "train" else ["original_val"]
        elif split == "val":
            targets = ["test"]
        else:
            targets = ["historical_dev"] if image["path"] in dev_paths else ["expanded_train"]
        for target in targets:
            destination = OUT / target / image["label"] / (source + "_" + Path(image["path"]).name)
            shutil.copy2(image["path"], destination)
        manifest.append(dict(**image, destinations=targets))

    # Save one reproducible camera-like version per evaluation image, before training.
    # This tests moderate geometric/lighting shifts, not actual phone-camera accuracy.
    torch.manual_seed(314159)
    camera = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(.7, 1.0)),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.2, hue=.03),
    ])
    for name in ["original_val", "historical_dev", "test"]:
        for label in CLASSES:
            for path in sorted((OUT / name / label).glob("*.jpg")):
                with Image.open(path) as image:
                    camera(image.convert("RGB")).save(OUT / ("camera_" + name) / label / (path.stem + ".png"))

    counts = {name: {label: len(list((OUT / name / label).glob("*"))) for label in CLASSES} for name in names}
    audit = dict(classes=CLASSES, split_seed=2026, camera_seed=314159,
                 input_images=sum(map(len, groups.values())), unique_pixel_groups=len(groups),
                 excluded_count=len(excluded), exclusion_reasons=dict(Counter(x["reason"] for x in excluded)),
                 counts=counts, excluded=excluded, manifest=manifest)
    (RESULTS / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in ["manifest", "excluded"]}, indent=2))


if __name__ == "__main__":
    main()
