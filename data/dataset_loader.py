"""
Clean interface between the federated learning system and whatever dataset gets used.

The rest of the codebase (partitioning, client training, evaluation) only ever talks to
this interface - never to a specific dataset's file format. When the real dataset is
ready, only this file needs a new subclass; nothing in federated/ changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetInfo:
    num_classes: int
    class_names: List[str]
    input_shape: Tuple[int, int, int]  # (channels, height, width)


class MedicalImageDataset(ABC):
    """Subclass this for any real dataset. Must return a torch Dataset of
    (image_tensor, label) pairs plus metadata describing the classification task."""

    @abstractmethod
    def load(self) -> Tuple[Dataset, Dataset, DatasetInfo]:
        """Returns (train_dataset, test_dataset, info)."""
        raise NotImplementedError


class SyntheticBrainMRIDataset(MedicalImageDataset):
    """
    Placeholder implementation so the whole pipeline is runnable and testable before
    the real dataset is wired in. Generates synthetic 'MRI-like' tensors with class-
    dependent statistical patterns (not random noise - each class has a distinct mean/
    texture pattern so a model can actually learn something), matching the shape of a
    typical brain tumor MRI classification task (4 classes, as in the existing
    ResNet50/DenseNet201/VGG16 brain tumor project).
    """

    def __init__(self, num_samples: int = 2000, image_size: int = 64, num_classes: int = 4, seed: int = 42):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.seed = seed

    def load(self) -> Tuple[Dataset, Dataset, DatasetInfo]:
        rng = np.random.default_rng(self.seed)
        images, labels = [], []
        for i in range(self.num_samples):
            label = i % self.num_classes
            base = rng.normal(loc=label * 0.15, scale=0.3, size=(1, self.image_size, self.image_size))
            texture = rng.normal(loc=0, scale=0.05 + 0.02 * label, size=(1, self.image_size, self.image_size))
            img = np.clip(base + texture, -1, 1).astype(np.float32)
            images.append(img)
            labels.append(label)

        images = np.stack(images)
        labels = np.array(labels)

        split = int(0.8 * self.num_samples)
        train_ds = _TensorImageDataset(images[:split], labels[:split])
        test_ds = _TensorImageDataset(images[split:], labels[split:])

        info = DatasetInfo(
            num_classes=self.num_classes,
            class_names=["glioma", "meningioma", "pituitary", "no_tumor"][: self.num_classes],
            input_shape=(1, self.image_size, self.image_size),
        )
        return train_ds, test_ds, info


class _TensorImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_dataset(name: str = "synthetic") -> MedicalImageDataset:
    """Factory - swap in the real dataset here once it's available, e.g.:
    if name == 'brain_mri_kaggle': return BrainMRIKaggleDataset(path=...)
    """
    if name == "synthetic":
        return SyntheticBrainMRIDataset()
    raise ValueError(f"Unknown dataset: {name}")
