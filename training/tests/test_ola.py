import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from ola import overlap_add


@pytest.mark.parametrize("B,C,n,hop,frames", [(1, 602, 600, 150, 37), (2, 152, 150, 75, 5), (1, 512, 1024, 256, 9), (1, 1, 600, 150, 1), (1, 22, 20, 5, 50)])
def test_matches_conv_transpose(B, C, n, hop, frames):
    x, basis = torch.randn(B, C, frames, dtype=torch.float64), torch.randn(C, 1, n, dtype=torch.float64)
    torch.testing.assert_close(overlap_add(x, basis, hop), F.conv_transpose1d(x, basis, stride=hop))


class InverseSTFT(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer("inv", torch.randn(602, 1, 600))

    def forward(self, x):
        return overlap_add(x, self.inv, 150)


def test_onnx_export_has_no_conv_transpose_and_dynamic_frames(tmp_path):
    m, path = InverseSTFT(), tmp_path / "ola.onnx"
    torch.onnx.export(m, (torch.randn(1, 602, 20),), path, input_names=["x"], dynamic_axes={"x": {2: "frames"}}, opset_version=17, dynamo=False)
    assert "ConvTranspose" not in {n.op_type for n in onnx.load(path).graph.node}
    x = torch.randn(1, 602, 57)
    y = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
    np.testing.assert_allclose(y, F.conv_transpose1d(x, m.inv, stride=150).numpy(), atol=1e-4)
