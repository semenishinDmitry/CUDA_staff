#include <cuda_runtime.h>
#include <cmath>

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ----------------------------
// softmax kernel
// ----------------------------
__global__ void softmax_kernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[]; // shared memory per block
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_max = -1e20f;

    //  find max using grid-stride loop
    for (int i = idx; i < N; i += stride)
        local_max = fmaxf(local_max, input[i]);

    // reduce max within warp
    local_max = warpReduceMax(local_max);

    // first thread of each warp writes to shared memory
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) sdata[warp_id] = local_max;
    __syncthreads();

    // reduce max across warps (thread 0)
    if (tid < 32) {
        local_max = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : -1e20f;
        local_max = warpReduceMax(local_max);
    }
    __shared__ float max_val_block;
    if (tid == 0) max_val_block = local_max;
    __syncthreads();

    // compute sum(exp(x - max))
    float sum = 0.0f;
    for (int i = idx; i < N; i += stride)
        sum += expf(input[i] - max_val_block);

    // reduce sum within warp
    sum = warpReduceSum(sum);
    if (lane == 0) sdata[warp_id] = sum;
    __syncthreads();

    // reduce sum across warps
    if (tid < 32) {
        sum = (tid < (blockDim.x + 31) / 32) ? sdata[tid] : 0.0f;
        sum = warpReduceSum(sum);
    }
    __shared__ float sum_exp_block;
    if (tid == 0) sum_exp_block = sum;
    __syncthreads();

    //  compute final softmax
    for (int i = idx; i < N; i += stride)
        output[i] = expf(input[i] - max_val_block) / sum_exp_block;
}

// ----------------------------
// host code stays the same
// ----------------------------
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, N);
    cudaDeviceSynchronize();
}