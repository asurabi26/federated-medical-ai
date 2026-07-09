import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from data.dataset_loader import SyntheticBrainMRIDataset
from federated.compression import TorchQuantizeCompressor, raw_size_bytes
from federated.model import SimpleMedicalCNN
from federated.partition import partition_non_iid


def test_synthetic_dataset_loads_with_correct_shapes():
    dataset = SyntheticBrainMRIDataset(num_samples=100, image_size=32, num_classes=4)
    train_ds, test_ds, info = dataset.load()

    assert len(train_ds) == 80
    assert len(test_ds) == 20
    assert info.num_classes == 4

    image, label = train_ds[0]
    assert image.shape == (1, 32, 32)
    assert 0 <= label.item() < 4


def test_non_iid_partition_produces_skewed_distributions():
    dataset = SyntheticBrainMRIDataset(num_samples=400, image_size=16, num_classes=4)
    train_ds, _, info = dataset.load()

    subsets = partition_non_iid(train_ds, num_clients=4, num_classes=info.num_classes, alpha=0.3)

    assert len(subsets) == 4
    total = sum(len(s) for s in subsets)
    assert total == len(train_ds)

    # with low alpha, at least one client should be noticeably skewed toward one class
    import numpy as np
    found_skew = False
    for subset in subsets:
        if len(subset) == 0:
            continue
        counts = np.zeros(info.num_classes, dtype=int)
        for i in range(len(subset)):
            _, label = subset[i]
            counts[label.item()] += 1
        if counts.max() / max(counts.sum(), 1) > 0.6:
            found_skew = True
    assert found_skew, "expected at least one client to be skewed toward a dominant class"


def test_model_forward_pass_shape():
    model = SimpleMedicalCNN(in_channels=1, num_classes=4)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 4)


def test_quantize_compressor_round_trip_reduces_size():
    compressor = TorchQuantizeCompressor()
    state_dict = {"weight": torch.randn(1000, 1000)}

    raw_bytes = raw_size_bytes(state_dict)
    compressed = compressor.compress(state_dict)
    compressed_bytes = compressor.compressed_size_bytes(compressed)

    assert compressed_bytes < raw_bytes
    assert compressed_bytes / raw_bytes < pytest_approx_ratio()

    decompressed = compressor.decompress(compressed)
    # lossy quantization - values should be close but not exact
    max_error = (state_dict["weight"] - decompressed["weight"]).abs().max().item()
    assert max_error < 0.1  # quantization error should be small for a normal-range tensor


def pytest_approx_ratio():
    return 0.3  # float32 (4 bytes) -> uint8 (1 byte) should be roughly a 4x reduction
