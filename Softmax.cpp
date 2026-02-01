#include <cmath>
#include <cuda_runtime.h>

// warp = 32 потока, которые:
// 	•	выполняются синхронно,
// 	•	имеют ID lane_id от 0 до 31,
// 	•	могут обмениваться регистрами без shared memory.
// •	fmaxf = корректный float max
// >>= 1 = деление на 2
// 0xffffffff = маска для всех 32 потоков в warp
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
// softmax kernel (single block)
// ----------------------------
__global__ void softmax_kernel(const float *input, float *output, int N) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;

  // ----------------------------
  // 1. find max
  // ----------------------------
  float local_max = -1e20f;
  for (int i = tid; i < N; i += blockDim.x)
    local_max = fmaxf(local_max, input[i]);

  local_max = warpReduceMax(local_max);

  if (lane == 0)
    sdata[warp_id] = local_max;
  __syncthreads();

  float max_val;
  if (warp_id == 0) {
    max_val = (tid < (blockDim.x + 31) / 32) ? sdata[lane] : -1e20f;
    max_val = warpReduceMax(max_val);
    if (tid == 0)
      sdata[0] = max_val;
  }
  __syncthreads();

  max_val = sdata[0];

  // ----------------------------
  // 2. compute sum(exp(x - max))
  // ----------------------------
  float sum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x)
    sum += expf(input[i] - max_val);

  sum = warpReduceSum(sum);

  if (lane == 0)
    sdata[warp_id] = sum;
  __syncthreads();

  float sum_val;
  if (warp_id == 0) {
    sum_val = (tid < (blockDim.x + 31) / 32) ? sdata[lane] : 0.0f;
    sum_val = warpReduceSum(sum_val);
    if (tid == 0)
      sdata[0] = sum_val;
  }
  __syncthreads();

  sum_val = sdata[0];

  // ----------------------------
  // 3. write output
  // ----------------------------
  for (int i = tid; i < N; i += blockDim.x)
    output[i] = expf(input[i] - max_val) / sum_val;
}

// ----------------------------
// host entry point
// ----------------------------
extern "C" void solve(const float *input, float *output, int N) {
  const int threadsPerBlock = 256;
  const int blocksPerGrid = 1;

  softmax_kernel<<<blocksPerGrid, threadsPerBlock,
                   threadsPerBlock * sizeof(float)>>>(input, output, N);

  cudaDeviceSynchronize();
}