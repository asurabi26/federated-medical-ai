"""
Builds the custom CUDA quantization kernel as a loadable PyTorch extension.

Run this in an environment with a CUDA-capable GPU and the CUDA toolkit installed
(e.g. Google Colab with a T4/A100 runtime, or any machine with an NVIDIA GPU + nvcc).
It will NOT build on a CPU-only machine - torch.utils.cpp_extension needs nvcc to
compile the .cu file.

Usage:
    python setup.py install
or, more commonly for quick iteration (used in the Colab notebook):
    from torch.utils.cpp_extension import load
    quant_cuda = load(name="quant_cuda", sources=["quantize_kernel.cu"], ...)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="quant_cuda",
    ext_modules=[
        CUDAExtension(
            name="quant_cuda",
            sources=["quantize_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
