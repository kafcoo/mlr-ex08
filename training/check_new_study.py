"""Validate the 21-run development-only study and its selection/protocol."""
import hashlib
import json
from pathlib import Path

import numpy as np

from train import ROOT, EVAL_SETS

RUN = Path("training/runs/networks_v1")


def main():
    protocol = json.loads((RUN / "protocol.json").read_text())
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    assert protocol["network_source_sha256"] == hashlib.sha256(Path("training/networks.py").read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for name in ["original_train", "expanded_train"] + EVAL_SETS:
        for path in sorted(p for p in (ROOT / name).rglob("*") if p.is_file()):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    assert digest.hexdigest() == protocol["data_sha256"]
    runs = [json.loads(path.read_text()) for path in (RUN / "results").glob("*_seed*.json")]
    assert len(runs) == 21 and len({run["variant"] for run in runs}) == 7
    for variant in {run["variant"] for run in runs}:
        assert {run["seed"] for run in runs if run["variant"] == variant} == {0, 1, 2}
    for run in runs:
        assert run["protocol_sha256"] == fingerprint and run["train_count"] == 649
        assert len(run["history"]) == run["epochs"] == 15
        assert run["best_dev_score"] == max(row["dev_score"] for row in run["history"])
        assert run["history"][run["best_epoch"]-1]["dev_score"] == run["best_dev_score"]
        rates = [row["learning_rate"] for row in run["history"]]
        if run["variant"] == "mobile_cosine":
            assert rates[0] == .001 and all(a > b for a, b in zip(rates, rates[1:]))
        else:
            assert all(rate == .001 for rate in rates)
        if run["variant"] == "mobile_frozen":
            assert run["trainable_parameters"] == 4100 and run["settings"]["frozen"]
        for row in run["history"]:
            assert set(row["metrics"]) == set(EVAL_SETS)
            assert abs(row["dev_score"] - np.mean([m["balanced_accuracy"] for m in row["metrics"].values()])) < 1e-8
    evaluation = json.loads((RUN / "results/evaluation.json").read_text())
    for model in evaluation["models"]:
        assert set(model["metrics"]) == set(EVAL_SETS), "Historical test must not appear in the new evaluation"
        for name, metric in model["metrics"].items():
            matrix = np.array(metric["confusion"])
            assert matrix.shape == (4, 4) and matrix.sum() == metric["count"]
            assert abs(np.trace(matrix)/matrix.sum() - metric["accuracy"]) < 1e-7
            assert abs(np.mean(matrix.diagonal()/matrix.sum(1)) - metric["balanced_accuracy"]) < 1e-7
    selection = evaluation["selection"]
    means = {variant: np.mean([r["best_dev_score"] for r in runs if r["variant"] == variant]) for variant in {r["variant"] for r in runs}}
    assert max(means, key=means.get) == selection["selected_variant"]
    candidates = sorted([r for r in runs if r["variant"] == selection["selected_variant"]], key=lambda r: (r["best_dev_score"], r["seed"]))
    assert candidates[1]["seed"] == selection["selected_seed"]
    result = dict(status="PASS", runs=21, seeds=[0, 1, 2], historical_test_used=False,
                  checks=["Actual train/dev and architecture source fingerprints", "7 variants × 3 seeds × 15 epochs",
                          "Same 649 training images and fixed development views", "Correct best epochs and LR schedules",
                          "Frozen-head parameter budget", "Confusion matrices and metrics", "Development-only mean/median selection"])
    (RUN / "results/integrity_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
