#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void dot_warp_reduce(const float *A, const float *B, float *result,
                                int N) {
  float sum = 0.0f;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = idx; i < N; i += stride)
    sum += A[i] * B[i];

  // warp-level reduction
  sum = warp_reduce_sum(sum);

  // one thread per warp writes to shared
  __shared__ float warp_sums[32]; // max 1024 / 32 = 32 warps
  int warp_id = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;

  if (lane == 0)
    warp_sums[warp_id] = sum;

  __syncthreads();

  // final reduction by first warp
  if (warp_id == 0) {
    sum = (threadIdx.x < blockDim.x / warpSize) ? warp_sums[lane] : 0.0f;

    sum = warp_reduce_sum(sum);

    if (lane == 0)
      atomicAdd(result, sum);
  }
}

extern "C" void solve(const float *A, const float *B, float *result, int N) {
  const int blockSize = 256;
  int numBlocks = min(1024, (N + blockSize - 1) / blockSize);

  cudaMemset(result, 0, sizeof(float));

  dot_warp_reduce<<<numBlocks, blockSize>>>(A, B, result, N);
}