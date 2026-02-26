#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <mma.h>

__device__ __forceinline__ int round_half_away_from_zero(float x) {
    return (x >= 0.f) ? (int)floorf(x + 0.5f)
                      : (int)ceilf(x - 0.5f);
}

__global__ void qgemm_kernel(const int8_t* A, const int8_t* B, int8_t* C,
                             int M, int N, int K,
                             float scale_A, float scale_B, float scale_C,
                             int zero_point_A, int zero_point_B, int zero_point_C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int32_t acc = 0;
    for (int k = 0; k < K; ++k) {
        int a = (int)A[row * K + k] - zero_point_A;
        int b = (int)B[k * N + col] - zero_point_B;
        acc += a * b;
    }

    float scale = (scale_A * scale_B) / scale_C;
    float y = fmaf((float)acc, scale, 0.0f); 

    int out = round_half_away_from_zero(y) + zero_point_C;

    // saturate to int8
    out = out < -128 ? -128 : (out > 127 ? 127 : out);
    C[row * N + col] = (int8_t)out;
}


extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C,
                      int M, int N, int K,
                      float scale_A, float scale_B, float scale_C,
                      int zero_point_A, int zero_point_B, int zero_point_C)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    qgemm_kernel<<<grid, block>>>(
        A, B, C,
        M, N, K,
        scale_A, scale_B, scale_C,
        zero_point_A, zero_point_B, zero_point_C
    );

    cudaDeviceSynchronize();
}