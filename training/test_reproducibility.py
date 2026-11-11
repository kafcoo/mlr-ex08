"""Small real training smoke test for resuming safely; archived runs are untouched."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    checks = []
    Path(".local-tools").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="resume-test-", dir=".local-tools") as temporary:
        run_dir = Path(temporary)
        command = [sys.executable, "training/train.py", "--run-dir", temporary,
                   "--variants", "original_only", "--seeds", "7", "--epochs", "1", "--device", "cpu"]
        first = subprocess.run(command, capture_output=True, text=True, check=True)
        checkpoint = run_dir / "checkpoints/original_only_seed7.pt"
        result_path = run_dir / "results/original_only_seed7.json"
        result = json.loads(result_path.read_text())
        assert checkpoint.exists() and result["epochs"] == 1 and result["protocol_sha256"]
        checks.append("Fresh run writes real checkpoint, metric and data/config fingerprint")
        timestamp = checkpoint.stat().st_mtime_ns
        second = subprocess.run(command, capture_output=True, text=True, check=True)
        assert "Already completed with matching" in second.stdout
        assert checkpoint.stat().st_mtime_ns == timestamp
        checks.append("Matching checkpoint/protocol safely resumes without overwriting")
        checkpoint.unlink()
        third = subprocess.run(command, capture_output=True, text=True, check=True)
        assert checkpoint.exists() and "Already completed" not in third.stdout
        checks.append("Published metrics without weights do not skip required training")
        changed = command.copy()
        changed[changed.index("--epochs") + 1] = "2"
        mismatch = subprocess.run(changed, capture_output=True, text=True)
        assert mismatch.returncode != 0 and "settings changed" in mismatch.stderr
        checks.append("Changed epoch/config is rejected in an existing experiment directory")
    for baseline, message in [(".local-tools/missing-baseline.onnx", "baseline is missing"),
                              ("leaf_model.onnx", "Baseline SHA-256 does not match")]:
        process = subprocess.run([sys.executable, "training/evaluate.py", "--baseline", baseline], capture_output=True, text=True)
        assert process.returncode != 0 and message in process.stderr
    checks.append("Missing or wrong baseline cannot silently substitute the deployed model")
    process = subprocess.run([sys.executable, "training/train.py", "--variants", "typo"], capture_output=True, text=True)
    assert process.returncode != 0 and "invalid choice" in process.stderr
    checks.append("Unknown variant names are rejected rather than mislabelled")
    process = subprocess.run([sys.executable, "training/train.py"], capture_output=True, text=True)
    assert process.returncode != 0 and "Archived results" in process.stderr
    checks.append("Archived 12-run results are protected from accidental overwrite")
    output = dict(status="PASS", checks=checks, scope="temporary 1-epoch CPU runs; original measurements preserved")
    Path("training/results/reproducibility_checks.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
