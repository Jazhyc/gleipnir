// Minimal counter-access canary, independent of PyTorch and vLLM.
#include <cuda_runtime.h>
#include <cstdio>

__global__ void counter_probe(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    x[i] = float(i) * 2.0f;
}

int main() {
    float* x = nullptr;
    cudaError_t error = cudaMalloc(&x, 4096 * sizeof(float));
    if (error != cudaSuccess) return 1;
    counter_probe<<<16, 256>>>(x);
    error = cudaDeviceSynchronize();
    std::printf("CUDA result: %s\n", cudaGetErrorString(error));
    cudaFree(x);
    return error == cudaSuccess ? 0 : 1;
}
