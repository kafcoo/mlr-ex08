"""Small behavioral checks for the copied Task 3 models and frozen-head experiment."""
import ast
import json
from pathlib import Path

import torch
from torch import nn

from networks import build_model


def main():
    torch.set_num_threads(4)
    notebook = json.loads(Path("ex08_student.ipynb").read_text())
    namespace = dict(nn=nn, torch=torch)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "class LeafNet(" not in source and "class LeafNetBN(" not in source:
            continue
        definitions = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook classes", "exec"), namespace)
    checks = []
    torch.manual_seed(123)
    sample = torch.randn(4, 3, 96, 96)
    for name, original in [("leafnet", "LeafNet"), ("leafnet_bn", "LeafNetBN")]:
        model = build_model(name).eval()
        notebook_model = namespace[original]().eval()
        notebook_model.load_state_dict(model.state_dict())
        with torch.inference_mode():
            torch.testing.assert_close(model(sample), notebook_model(sample), atol=1e-7, rtol=1e-6)
        checks.append(f"{name}: same state and outputs as actual Task 3 notebook class")
    gap = build_model("leafnet_bn_gap").eval()
    with torch.inference_mode():
        assert gap(sample).shape == (4, 4)
    assert sum(p.numel() for p in gap.parameters()) < sum(p.numel() for p in build_model("leafnet_bn").parameters())
    checks.append("GAP preserves four logits while reducing parameter count")
    model = build_model("mobilenet_v3_small", pretrained=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier[3].parameters():
        parameter.requires_grad = True
    model.eval()
    model.classifier[3].train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001)
    loss = nn.functional.cross_entropy(model(sample), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    optimizer.step()
    changed = [key for key, value in model.state_dict().items() if not torch.equal(value, before[key])]
    assert changed and all(key.startswith("classifier.3.") for key in changed), changed
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 4100
    checks.append("Frozen experiment updates only 4100 last-layer parameters; all BN statistics remain unchanged")
    output = dict(status="PASS", checks=checks)
    Path("training/runs/networks_v1/results/network_checks.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
