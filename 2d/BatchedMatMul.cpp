#include <cuda_runtime.h>

#define TILE 16

__global__ void batched_gemm_naive(
    const float* A, const float* B, float* C,
    int BATCH, int M, int N, int K
) {
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < BATCH && row < M && col < N) {
        float sum = 0.0f;
        int A_base = b * M * K;
        int B_base = b * K * N;
        int C_base = b * M * N;

        for (int k = 0; k < K; ++k) {
            sum += A[A_base + row * K + k] *
                   B[B_base + k * N + col];
        }

        C[C_base + row * N + col] = sum;
    }
}

__global__ void batched_gemm_tiled(
    const float* A, const float* B, float* C,
    int BATCH, int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int b = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    int A_base = b * M * K;
    int B_base = b * K * N;
    int C_base = b * M * N;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int kA = t * TILE + threadIdx.x;
        int kB = t * TILE + threadIdx.y;

        if (row < M && kA < K)
            As[threadIdx.y][threadIdx.x] =
                A[A_base + row * K + kA];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (kB < K && col < N)
            Bs[threadIdx.y][threadIdx.x] =
                B[B_base + kB * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[C_base + row * N + col] = acc;
}

extern "C" void solve(const float* A, const float* B, float* C,
                      int BATCH, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE,
        BATCH
    );

    batched_gemm_tiled<<<grid, block>>>(A, B, C, BATCH, M, N, K);
}