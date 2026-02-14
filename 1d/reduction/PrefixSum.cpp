#include <cuda_runtime.h>

__device__ __forceinline__ float warp_scan(float val)
{
    // Inclusive scan inside warp
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % WARP_SIZE >= offset)
            val += n;
    }
    return val;
}

__global__ void block_prefix_sum_warp(const float* input, float* output, float* blockSums, int N)
{
    __shared__ float warpSums[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float val = (gid < N) ? input[gid] : 0.0f;

    // 1. Scan внутри warp
    val = warp_scan(val);

    int warpId = tid / WARP_SIZE;
    int lane   = tid % WARP_SIZE;

    // 2. Последний поток warp сохраняет сумму warp
    if (lane == WARP_SIZE - 1) {
        warpSums[warpId] = val;
    }

    __syncthreads();

    // 3. Первый warp сканирует warpSums
    if (warpId == 0) {
        float warpVal = (tid < BLOCK_SIZE / WARP_SIZE) ? warpSums[lane] : 0.0f;
        warpVal = warp_scan(warpVal);
        if (tid < BLOCK_SIZE / WARP_SIZE)
            warpSums[lane] = warpVal;
    }

    __syncthreads();

    // 4. Добавляем offset warp'а
    if (warpId > 0) {
        val += warpSums[warpId - 1];
    }

    if (gid < N)
        output[gid] = val;

    if (tid == blockDim.x - 1 && blockSums) {
        blockSums[blockIdx.x] = val;
    }
}

__global__ void add_block_offsets(float* output, const float* blockOffsets, int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N && blockIdx.x > 0) {
        output[gid] += blockOffsets[blockIdx.x - 1];
    }
}

extern "C" void solve(const float* input, float* output, int N)
{
    const int BLOCK_SIZE 256;
    const int WARP_SIZE 32
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* d_blockSums = nullptr;
    float* d_blockOffsets = nullptr;

    if (numBlocks > 1) {
        cudaMalloc(&d_blockSums, numBlocks * sizeof(float));
        cudaMalloc(&d_blockOffsets, numBlocks * sizeof(float));
    }

    block_prefix_sum_warp<<<numBlocks, BLOCK_SIZE>>>(input, output, d_blockSums, N);

    if (numBlocks > 1) {
        // prefix по blockSums (можно на GPU, но для краткости — на CPU)
        std::vector<float> h(numBlocks);
        cudaMemcpy(h.data(), d_blockSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 1; i < numBlocks; ++i)
            h[i] += h[i - 1];

        cudaMemcpy(d_blockOffsets, h.data(), numBlocks * sizeof(float), cudaMemcpyHostToDevice);

        add_block_offsets<<<numBlocks, BLOCK_SIZE>>>(output, d_blockOffsets, N);
    }

    cudaFree(d_blockSums);
    cudaFree(d_blockOffsets);
}