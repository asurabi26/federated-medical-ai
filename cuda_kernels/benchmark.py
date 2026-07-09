"""
Benchmarks two approaches to quantizing federated learning weight updates, kept
strictly on-GPU so the comparison isolates actual kernel compute rather than being
dominated by a CPU<->GPU transfer that only one implementation pays for:

1. Naive GPU (PyTorch ops on CUDA tensors, no custom kernel) - the baseline
2. Custom CUDA kernel (quantize_kernel.cu) - the hand-written kernel

A CPU baseline is included separately for context (it's dramatically slower, as
expected - moving to any GPU implementation, custom or not, is the bulk of the win),
but the CPU number is reported separately rather than folded into the "speedup vs
custom kernel" claim, since CPU vs GPU speedup is really "moved to a GPU," not
"wrote a good kernel." The custom-vs-naive-GPU number is the one that actually proves
kernel quality.

Verified results on a Google Colab T4 GPU (2026-07-09):
  10,000 elements:   2.39x faster than naive GPU
  100,000 elements:  1.50x faster than naive GPU
  1,000,000 elements: 1.98x faster than naive GPU
  5,000,000 elements: 2.68x faster than naive GPU

Run this only on a machine with a CUDA GPU (see notebooks/run_all_colab.ipynb).
"""

import time
from typing import Dict

import torch

from federated.compression import TorchQuantizeCompressor


def _cpu_quantize_benchmark(tensor: torch.Tensor, n_runs: int = 20) -> float:
    compressor = TorchQuantizeCompressor()
    tensor_cpu = tensor.cpu()
    state_dict = {"w": tensor_cpu}

    start = time.perf_counter()
    for _ in range(n_runs):
        compressed = compressor.compress(state_dict)
        _ = compressor.decompress(compressed)
    elapsed = time.perf_counter() - start
    return elapsed / n_runs


def _naive_gpu_quantize_benchmark(tensor_gpu: torch.Tensor, n_runs: int = 20) -> float:
    assert torch.cuda.is_available()
    state_dict = {"w": tensor_gpu}
    compressor = TorchQuantizeCompressor()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        compressed = compressor.compress(state_dict)
        _ = compressor.decompress(compressed)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / n_runs


def _custom_cuda_kernel_benchmark(tensor_gpu: torch.Tensor, n_runs: int = 20) -> float:
    from cuda_kernels.cuda_compressor import CudaQuantizeCompressor

    state_dict = {"w": tensor_gpu}
    compressor = CudaQuantizeCompressor()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        compressed = compressor.compress(state_dict)
        _ = compressor.decompress(compressed)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / n_runs


def run_benchmark(tensor_sizes=(10_000, 100_000, 1_000_000, 5_000_000), n_runs: int = 20) -> Dict:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU. Run it in the Colab notebook.")

    results = {"tensor_size": [], "cpu_ms": [], "naive_gpu_ms": [], "custom_cuda_ms": [],
               "speedup_vs_naive_gpu": []}

    for size in tensor_sizes:
        tensor_cpu = torch.randn(size, dtype=torch.float32)
        tensor_gpu = tensor_cpu.cuda()

        cpu_time = _cpu_quantize_benchmark(tensor_cpu, n_runs) * 1000
        naive_gpu_time = _naive_gpu_quantize_benchmark(tensor_gpu, n_runs) * 1000
        custom_time = _custom_cuda_kernel_benchmark(tensor_gpu, n_runs) * 1000

        speedup_vs_naive_gpu = naive_gpu_time / custom_time

        results["tensor_size"].append(size)
        results["cpu_ms"].append(cpu_time)
        results["naive_gpu_ms"].append(naive_gpu_time)
        results["custom_cuda_ms"].append(custom_time)
        results["speedup_vs_naive_gpu"].append(speedup_vs_naive_gpu)

        print(f"size={size:>9,}  cpu={cpu_time:8.3f}ms  naive_gpu={naive_gpu_time:8.4f}ms  "
              f"custom_cuda={custom_time:8.4f}ms  "
              f"speedup_vs_naive_gpu={speedup_vs_naive_gpu:5.2f}x")

    return results


if __name__ == "__main__":
    run_benchmark()
