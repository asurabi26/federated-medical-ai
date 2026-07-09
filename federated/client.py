"""
One simulated hospital node. Trains locally on its own private data subset for a few
epochs, then returns only its model weights (never the underlying images) - that's the
entire privacy premise of federated learning.
"""

import copy
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


class FederatedClient:
    def __init__(self, client_id: int, dataset: Subset, device: str = "cpu"):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device

    def local_train(self, global_weights: Dict[str, torch.Tensor], model_fn,
                     local_epochs: int = 1, lr: float = 0.01, batch_size: int = 16) -> Dict[str, torch.Tensor]:
        model = model_fn().to(self.device)
        model.load_state_dict(copy.deepcopy(global_weights))
        model.train()

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for _ in range(local_epochs):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    def __len__(self):
        return len(self.dataset)
