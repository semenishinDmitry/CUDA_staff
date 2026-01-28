#include <cuda_runtime.h>

constexpr int TILE = 16;

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
        output[col * rows + row] = input[row * cols + col];
}

__global__ void better_matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE][TILE+1]; // +1 против bank conflict
    
    const int x = blockIdx.y * TILE + threadIdx.y;
    const int y = blockIdx.x * TILE + threadIdx.x;

    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < rows && y < cols)
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
