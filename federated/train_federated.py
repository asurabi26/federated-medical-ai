"""
Orchestrates the full federated training process: partitions data across simulated
hospitals, runs local training rounds, aggregates via FedAvg, and tracks both accuracy
and communication cost per round - so we can report not just "does it converge" but
"how much bandwidth did convergence cost, and how much did compression save."
"""

import time
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from federated.client import FederatedClient
from federated.compression import TorchQuantizeCompressor, raw_size_bytes
from federated.model import SimpleMedicalCNN
from federated.partition import partition_non_iid, summarize_partition
from federated.server import FedAvgServer


def evaluate(model: torch.nn.Module, test_dataset, device: str = "cpu") -> float:
    model.eval()
    loader = DataLoader(test_dataset, batch_size=32)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def run_federated_training(
    train_dataset,
    test_dataset,
    num_classes: int,
    input_channels: int,
    num_clients: int = 5,
    num_rounds: int = 10,
    local_epochs: int = 1,
    non_iid_alpha: float = 0.5,
    use_compression: bool = True,
    device: str = "cpu",
    log_fn=print,
) -> dict:
    def model_fn():
        return SimpleMedicalCNN(in_channels=input_channels, num_classes=num_classes)

    client_subsets = partition_non_iid(train_dataset, num_clients, num_classes, alpha=non_iid_alpha)
    log_fn("Non-IID partition across simulated hospitals:")
    log_fn(summarize_partition(client_subsets, num_classes))

    clients = [FederatedClient(i, subset, device=device) for i, subset in enumerate(client_subsets)]
    compressor = TorchQuantizeCompressor() if use_compression else None
    server = FedAvgServer(compressor=compressor)

    global_model = model_fn().to(device)
    global_weights = {k: v.cpu() for k, v in global_model.state_dict().items()}

    history = {"round": [], "accuracy": [], "round_bytes": [], "raw_bytes": [], "round_seconds": []}

    for round_num in range(1, num_rounds + 1):
        t0 = time.time()
        client_weights, client_sizes = [], []
        for client in clients:
            if len(client) == 0:
                continue
            w = client.local_train(global_weights, model_fn, local_epochs=local_epochs)
            client_weights.append(w)
            client_sizes.append(len(client))

        raw_bytes = sum(raw_size_bytes(w) for w in client_weights)
        global_weights = server.aggregate(client_weights, client_sizes)

        global_model.load_state_dict(global_weights)
        global_model.to(device)
        acc = evaluate(global_model, test_dataset, device=device)
        elapsed = time.time() - t0

        history["round"].append(round_num)
        history["accuracy"].append(acc)
        history["round_bytes"].append(server.round_communication_bytes[-1])
        history["raw_bytes"].append(raw_bytes)
        history["round_seconds"].append(elapsed)

        compression_ratio = raw_bytes / server.round_communication_bytes[-1] if server.round_communication_bytes[-1] else 1.0
        log_fn(
            f"[round {round_num}/{num_rounds}] accuracy={acc:.3f} "
            f"comm={server.round_communication_bytes[-1] / 1024:.1f}KB "
            f"(raw={raw_bytes / 1024:.1f}KB, ratio={compression_ratio:.2f}x) "
            f"time={elapsed:.1f}s"
        )

    return {"model": global_model, "history": history}


if __name__ == "__main__":
    from data.dataset_loader import get_dataset

    dataset = get_dataset("synthetic")
    train_ds, test_ds, info = dataset.load()

    result = run_federated_training(
        train_ds, test_ds,
        num_classes=info.num_classes,
        input_channels=info.input_shape[0],
        num_clients=5,
        num_rounds=8,
    )
    print("Final accuracy:", result["history"]["accuracy"][-1])
