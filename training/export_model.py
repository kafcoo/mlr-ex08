"""Export the already-selected checkpoint and regenerate its reference probabilities."""
import hashlib
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

from train import MEAN, STD, RESULTS, CHECKPOINTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="training")
    parser.add_argument("--output-dir", default=".", help="Export into a separate folder to keep the deployed assets intact")
    args = parser.parse_args()
    results, checkpoints = Path(args.run_dir) / "results", Path(args.run_dir) / "checkpoints"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads((results / "selection.json").read_text())
    if not selection["deploy_candidate"]:
        raise SystemExit("No candidate improves development score; keep the original model.")
    torch.set_num_threads(4)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    model.load_state_dict(torch.load(checkpoints / selection["selected_checkpoint"], weights_only=True, map_location="cpu"))
    model.eval()
    temporary = checkpoints / "candidate.onnx"
    # Legacy exporter produces a compact, broadly supported opset without external data.
    torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), str(temporary),
                      input_names=["image"], output_names=["logits"], opset_version=17,
                      dynamo=False, external_data=False)
    classes = ["apple", "cherry", "chestnut", "maple"]
    model_info = dict(classes=classes, input_size=96, mean=MEAN, std=STD)
    onnx_model = onnx.load(str(temporary))
    onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(temporary))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(temporary), sess_options=options, providers=["CPUExecutionProvider"])
    normalise = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    sample = normalise(Image.open("check_image.jpg").convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        reference = model(sample).softmax(1).numpy()[0]
    output = session.run(None, {"image": sample.numpy()})[0][0]
    output = np.exp(output - output.max())
    output /= output.sum()
    np.testing.assert_allclose(reference, output, rtol=1e-4, atol=1e-6)
    # Also verify different inputs, not just a single easy reference image.
    generator = torch.Generator().manual_seed(987)
    max_error = 0.
    for _ in range(5):
        x = torch.randn(1, 3, 96, 96, generator=generator)
        with torch.inference_mode():
            expected = model(x).numpy()
        actual = session.run(None, {"image": x.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    (output_dir / "leaf_model.onnx").write_bytes(temporary.read_bytes())
    (output_dir / "model_info.json").write_text(json.dumps(model_info) + "\n")
    (output_dir / "check_expected.json").write_text(json.dumps(dict(zip(classes, map(float, reference))), indent=2) + "\n")
    if (output_dir / "check_image.jpg").resolve() != Path("check_image.jpg").resolve():
        shutil.copy2("check_image.jpg", output_dir / "check_image.jpg")
    with zipfile.ZipFile(output_dir / "ex08_for_agent.zip", "w") as archive:
        for name in ["leaf_model.onnx", "model_info.json", "check_image.jpg", "check_expected.json"]:
            archive.write(output_dir / name, name)
    # Lossless RGB fixture isolates browser JPEG decoding from model arithmetic.
    image = Image.open("check_image.jpg").convert("RGB")
    Image.frombytes("RGB", image.size, image.tobytes()).save(results / "reference_pixels.png")
    export = dict(variant=selection["selected_variant"], seed=selection["selected_seed"],
                  model_sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(), model_bytes=temporary.stat().st_size,
                  opset=17, max_logit_error=max_error, reference_probabilities=dict(zip(classes, map(float, reference))))
    (results / "export.json").write_text(json.dumps(export, indent=2) + "\n")
    print(json.dumps(export, indent=2))


if __name__ == "__main__":
    main()
