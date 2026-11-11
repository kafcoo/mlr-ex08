"""Export representative candidates for a fair browser timing/parity check."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from torchvision import transforms

from networks import build_model
from train import MEAN, STD

RUN = Path("training/runs/networks_v1")


def main():
    torch.set_num_threads(4)
    runs = [json.loads(path.read_text()) for path in sorted((RUN / "results").glob("*_seed*.json"))]
    assert len(runs) == 21, "Complete all seven variants × three seeds first"
    x = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])(Image.open("check_image.jpg").convert("RGB")).unsqueeze(0)
    profiles = []
    for variant in sorted({run["variant"] for run in runs}):
        candidates = sorted([run for run in runs if run["variant"] == variant], key=lambda run: (run["best_dev_score"], run["seed"]))
        representative = candidates[1]
        architecture = representative["architecture"]
        model = build_model(architecture).eval()
        model.load_state_dict(torch.load(RUN / "checkpoints" / f"{variant}_seed{representative['seed']}.pt", weights_only=True, map_location="cpu"))
        path = RUN / "checkpoints" / f"profile_{variant}.onnx"
        torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), str(path), input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False, external_data=False)
        graph = onnx.load(str(path))
        graph.metadata_props.add(key="model_info", value=json.dumps(dict(classes=representative["classes"], input_size=96, mean=MEAN, std=STD)))
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            expected = model(x).softmax(1).numpy()[0]
        logits = session.run(None, {"image": x.numpy()})[0][0]
        actual = np.exp(logits - logits.max())
        actual /= actual.sum()
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        profiles.append(dict(variant=variant, architecture=architecture, representative_seed=representative["seed"],
                             mean_dev_score=float(np.mean([r["best_dev_score"] for r in candidates])),
                             std_dev_score=float(np.std([r["best_dev_score"] for r in candidates])),
                             parameters=representative["parameter_count"], trainable_parameters=representative["trainable_parameters"],
                             mean_training_seconds=float(np.mean([r["seconds"] for r in candidates])),
                             onnx_bytes=path.stat().st_size, onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             local_model_path=str(path), reference_probabilities=expected.tolist()))
    (RUN / "results/profiles.json").write_text(json.dumps(profiles, indent=2) + "\n")
    print(json.dumps(profiles, indent=2))


if __name__ == "__main__":
    main()
