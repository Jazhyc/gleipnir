// Native signed INT4 operands, INT32 accumulation. Capability test, not GEMM timing.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

__global__ void native_int4(int* output, unsigned a, unsigned b) {
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b));
    int offset = threadIdx.x * 4;
    output[offset] = c0;
    output[offset + 1] = c1;
    output[offset + 2] = c2;
    output[offset + 3] = c3;
}

int main() {
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s SM%d%d\n", prop.name, prop.major, prop.minor);
    int* device;
    check(cudaMalloc(&device, 128 * sizeof(int)));
    // Constant operands test positive, negative, boundary, and zero values.
    const int pairs[][2] = {{1, 1}, {-1, 2}, {-8, 7}, {-8, -8}, {0, 7}};
    for (auto& pair : pairs) {
        unsigned a = static_cast<unsigned>(pair[0] & 15) * 0x11111111u;
        unsigned b = static_cast<unsigned>(pair[1] & 15) * 0x11111111u;
        native_int4<<<1, 32>>>(device, a, b);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        int host[128];
        check(cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost));
        int expected = 64 * pair[0] * pair[1];
        for (int value : host) {
            if (value != expected) {
                std::fprintf(stderr, "FAIL: got %d expected %d\n", value, expected);
                return 2;
            }
        }
        std::printf("PASS: 128 outputs equal %d (64 * %d * %d)\n",
                    expected, pair[0], pair[1]);
    }
    check(cudaFree(device));
}
