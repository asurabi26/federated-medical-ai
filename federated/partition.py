"""
Partitions a dataset across N simulated hospital nodes in a deliberately non-IID way -
this is the realistic and hard part of federated learning. Real hospitals don't see
identically distributed patient populations: a cancer center sees more of certain tumor
types, a rural clinic sees a different case mix. A uniform random split would make
federated learning trivially easy and wouldn't demonstrate anything the interview
question actually probes.

Uses a Dirichlet distribution to control the skew - a standard technique in federated
learning research (used in the original FedAvg follow-up papers) for simulating
realistic non-IID client data.
"""

from typing import List

import numpy as np
from torch.utils.data import Dataset, Subset


def partition_non_iid(dataset: Dataset, num_clients: int, num_classes: int,
                       alpha: float = 0.5, seed: int = 42) -> List[Subset]:
    """
    alpha controls the skew: low alpha (e.g. 0.1) = highly skewed, each hospital sees
    mostly one or two classes. High alpha (e.g. 10) = nearly uniform/IID. 0.5 is a
    commonly used realistic middle ground in federated learning literature.
    """
    rng = np.random.default_rng(seed)

    labels = np.array([dataset[i][1].item() if hasattr(dataset[i][1], "item") else dataset[i][1]
                        for i in range(len(dataset))])

    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    for idx_list in class_indices:
        rng.shuffle(idx_list)

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        splits = np.split(class_indices[c], proportions)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    for idx_list in client_indices:
        rng.shuffle(idx_list)

    return [Subset(dataset, idx_list) for idx_list in client_indices]


def summarize_partition(subsets: List[Subset], num_classes: int) -> str:
    """Human-readable class distribution per client, useful for the dashboard/README
    to show the non-IID skew is real and not accidentally uniform."""
    lines = []
    for client_id, subset in enumerate(subsets):
        counts = np.zeros(num_classes, dtype=int)
        for i in range(len(subset)):
            _, label = subset[i]
            label = label.item() if hasattr(label, "item") else label
            counts[label] += 1
        lines.append(f"hospital_{client_id}: {counts.tolist()} (total={len(subset)})")
    return "\n".join(lines)
