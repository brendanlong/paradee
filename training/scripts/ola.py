"""Overlap-add for the exported inverse STFTs."""
import torch.nn.functional as F


def overlap_add(x, basis, hop):
    """F.conv_transpose1d(x, basis, stride=hop) for x [B,C,frames] and basis [C,1,n] with n a multiple of hop, written as
    a MatMul and a sum of shifted slices. ONNX Runtime's CUDA EP is very slow at the ConvTranspose form of this shape
    (one output channel, kernel n_fft); on CPU the two are the same speed."""
    C, _, n = basis.shape; R = n // hop; B = x.shape[0]
    assert n % hop == 0, (n, hop)
    frames = (x.transpose(1, 2) @ basis.reshape(C, n)).reshape(B, -1, R, hop)          # [B, frames, R, hop]
    y = sum(F.pad(frames[:, :, r], (0, 0, r, R - 1 - r)) for r in range(R))             # [B, frames + R - 1, hop]
    return y.reshape(B, 1, -1)
