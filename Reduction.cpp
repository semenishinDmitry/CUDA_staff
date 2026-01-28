#include <cuda_runtime.h>

// input:  [x0, x1, x2, x3, x4, x5, x6, x7, ... xN]
// let blockDim.x = 8
// gridDim.x = 2

// Grid-stride loop: global memory -> registers
//  Block 0: threads 0 1 2 3 4 5 6 7
//  Block 1: threads 8 9 A B C D E F

// stride = blockDim * gridDim = 16
// tid 0: input[0], input[16], input[32], ...
// tid 1: input[1], input[17], input[33], ...

// In registers:
// sum (thread 0) = x0 + x16 + x32 + ...
// sum (thread 1) = x1 + x17 + x33 + ...

// Registers -> Shared
//  Block 0, sdata:
//  ┌────┬────┬────┬────┬────┬────┬────┬────┐
//  │ s0 │ s1 │ s2 │ s3 │ s4 │ s5 │ s6 │ s7 │
//  └────┴────┴────┴────┴────┴────┴────┴────┘
//  s0 = sum(thread 0)
//  s1 = sum(thread 1)

// Reduction in shared memory
//  Step 1: s0 += s4, s1 += s5, s2
//  Step 2: s0 += s2, s1 += s3
//  We get the sum of the block in s0

// Warp-level reduction (no __syncthreads())
// Use __shfl_down_sync to reduce within a warp

__global__ void reduce_stage1(const float *input, float *partial, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  float sum = 0.0f;

  // grid-stride loop
  for (int i = idx; i < N; i += stride)
    sum += input[i];

  sdata[tid] = sum;
  __syncthreads();

  // block reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    partial[blockIdx.x] = sdata[0];
}

__global__ void reduce_stage2(const float *partial, float *output, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;

  if (tid < N)
    sdata[tid] = partial[tid];
  else
    sdata[tid] = 0.0f;

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    output[0] = sdata[0];
}

extern "C" void solve(const float *input, float *output, int N) {
  const int blockSize = 256;
  int numBlocks = min(1024, (N + blockSize - 1) / blockSize);

  float *d_partial;
  cudaMalloc(&d_partial, numBlocks * sizeof(float));

  // stage 1
  reduce_stage1<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
      input, d_partial, N);

  // stage 2 (1 block)
  int finalBlock = 1;
  while (finalBlock < numBlocks)
    finalBlock <<= 1;

  reduce_stage2<<<1, finalBlock, finalBlock * sizeof(float)>>>(
      d_partial, output, numBlocks);

  cudaFree(d_partial);
}