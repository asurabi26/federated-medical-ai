"""
Central server: runs FedAvg (Federated Averaging, McMahan et al. 2017) - the standard
aggregation algorithm for federated learning. Weighted by each client's local dataset
size, since a hospital with more patients should influence the global model more.

The `compressor` hook is where the CUDA quantization kernel plugs in (see
cuda_kernels/). Each client's weights get compressed before being "sent" to the server
and decompressed on arrival - simulating the real communication step and letting us
measure the actual bytes saved, not just claim a speedup.
"""

from typing import Callable, Dict, List, Optional

import torch


class FedAvgServer:
    def __init__(self, compressor: Optional[Callable] = None):
        """compressor, if provided, must expose .compress(state_dict) -> compressed_blob
        and .decompress(compressed_blob) -> state_dict, and .stats for size reporting."""
        self.compressor = compressor
        self.round_communication_bytes = []

    def aggregate(self, client_weights: List[Dict[str, torch.Tensor]],
                  client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        total_size = sum(client_sizes)
        avg_weights = {}

        round_bytes = 0
        processed_client_weights = []

        for weights in client_weights:
            if self.compressor is not None:
                compressed = self.compressor.compress(weights)
                round_bytes += self.compressor.compressed_size_bytes(compressed)
                weights = self.compressor.decompress(compressed)
            else:
                round_bytes += sum(t.numel() * t.element_size() for t in weights.values())
            processed_client_weights.append(weights)

        self.round_communication_bytes.append(round_bytes)

        for key in processed_client_weights[0].keys():
            if processed_client_weights[0][key].dtype.is_floating_point:
                avg_weights[key] = sum(
                    w[key] * (size / total_size) for w, size in zip(processed_client_weights, client_sizes)
                )
            else:
                # non-float buffers (e.g. batchnorm num_batches_tracked) - just take client 0's
                avg_weights[key] = processed_client_weights[0][key]

        return avg_weights
