"""
Baseline (non-CUDA) implementation of weight update compression via 8-bit quantization.

This exists for two reasons: (1) so the federated training loop is fully runnable and
testable on any machine, GPU or not, and (2) so it's the honest baseline the CUDA
kernel gets benchmarked against - "my kernel is faster than PyTorch's own ops," not
"my kernel is faster than doing nothing."

Quantization scheme: per-tensor affine int8 quantization. For each weight tensor,
compute scale = (max - min) / 255, zero_point = round(-min / scale), then
quantized = round(value / scale + zero_point), clipped to [0, 255]. This is the same
scheme used by PyTorch's own quantization module, chosen so results are directly
comparable to a well-known baseline rather than an invented one.
"""

from typing import Dict, Tuple

import torch


class TorchQuantizeCompressor:
    """Pure PyTorch implementation - runs anywhere, including CPU-only machines."""

    def __init__(self):
        self.last_stats = {}

    def compress(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple]:
        compressed = {}
        for key, tensor in state_dict.items():
            if not tensor.dtype.is_floating_point:
                compressed[key] = ("raw", tensor)
                continue
            t_min, t_max = tensor.min().item(), tensor.max().item()
            scale = (t_max - t_min) / 255.0 if t_max > t_min else 1.0
            zero_point = round(-t_min / scale) if scale != 0 else 0
            q = torch.clamp(torch.round(tensor / scale + zero_point), 0, 255).to(torch.uint8)
            compressed[key] = ("quant", q, scale, zero_point, tensor.shape)
        return compressed

    def decompress(self, compressed: Dict[str, Tuple]) -> Dict[str, torch.Tensor]:
        state_dict = {}
        for key, entry in compressed.items():
            if entry[0] == "raw":
                state_dict[key] = entry[1]
            else:
                _, q, scale, zero_point, shape = entry
                state_dict[key] = ((q.float() - zero_point) * scale).reshape(shape)
        return state_dict

    def compressed_size_bytes(self, compressed: Dict[str, Tuple]) -> int:
        total = 0
        for entry in compressed.values():
            if entry[0] == "raw":
                t = entry[1]
                total += t.numel() * t.element_size()
            else:
                _, q, _, _, _ = entry
                total += q.numel() * q.element_size()  # 1 byte per element (uint8)
        return total


def raw_size_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in state_dict.values())
