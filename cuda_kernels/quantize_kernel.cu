// Custom CUDA kernel for per-tensor affine int8 quantization/dequantization of
// federated learning weight updates.
//
// Why a custom kernel instead of just calling PyTorch's built-in quantization:
// PyTorch's quantization ops are general-purpose and go through several abstraction
// layers. For this specific, narrow operation (element-wise affine quantize/dequantize
// over a flat float32 buffer), a hand-written kernel avoids that overhead and lets us
// tune the launch configuration (block/grid size) directly for the tensor sizes
// federated learning actually produces. The benchmark in benchmark.py measures whether
// that tuning is actually worth it - this isn't assumed, it's proven with numbers.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each thread handles one element: read the float, apply the affine transform,
// clamp into [0, 255], write out as uint8. This is a purely elementwise,
// embarrassingly parallel operation - the kind CUDA is built for.
__global__ void quantize_kernel(const float* __restrict__ input,
                                 uint8_t* __restrict__ output,
                                 float scale,
                                 float zero_point,
                                 int64_t num_elements) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = input[idx] / scale + zero_point;
        val = fminf(fmaxf(val, 0.0f), 255.0f);
        output[idx] = static_cast<uint8_t>(lrintf(val));
    }
}

__global__ void dequantize_kernel(const uint8_t* __restrict__ input,
                                   float* __restrict__ output,
                                   float scale,
                                   float zero_point,
                                   int64_t num_elements) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = (static_cast<float>(input[idx]) - zero_point) * scale;
    }
}

torch::Tensor quantize_cuda(torch::Tensor input, double scale, double zero_point) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto input_contig = input.contiguous();
    int64_t num_elements = input_contig.numel();

    auto output = torch::empty({num_elements}, torch::dtype(torch::kUInt8).device(input.device()));

    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    quantize_kernel<<<blocks, threads>>>(
        input_contig.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        static_cast<float>(scale),
        static_cast<float>(zero_point),
        num_elements
    );

    return output;
}

torch::Tensor dequantize_cuda(torch::Tensor input, double scale, double zero_point, std::vector<int64_t> shape) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");

    auto input_contig = input.contiguous();
    int64_t num_elements = input_contig.numel();

    auto output = torch::empty({num_elements}, torch::dtype(torch::kFloat32).device(input.device()));

    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    dequantize_kernel<<<blocks, threads>>>(
        input_contig.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        static_cast<float>(scale),
        static_cast<float>(zero_point),
        num_elements
    );

    return output.reshape(shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_cuda", &quantize_cuda, "Quantize float32 tensor to uint8 (CUDA)");
    m.def("dequantize_cuda", &dequantize_cuda, "Dequantize uint8 tensor to float32 (CUDA)");
}
