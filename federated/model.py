"""
A lightweight CNN for the federated image classification task.

Deliberately small: federated learning already means N nodes each training a full
copy of the model, so keeping it lightweight matters (and it lets rounds run in
reasonable time on Colab's free T4 in the demo). Swap in a real transfer-learning
backbone (ResNet50/DenseNet201 as used in the earlier brain tumor project) once
training on real data at real scale - the training loop doesn't care which model
class it's holding.
"""

import torch
import torch.nn as nn


class SimpleMedicalCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
