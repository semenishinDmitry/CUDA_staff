#include <cuda_runtime.h>

constexpr int TILE_WIDTH = 16;

__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < K) {
    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
      // каждый поток перечитывает A and B из global memory
      // кэш плохо работает в этом случае
      sum += A[row * N + k] * B[k * K + col];

    C[row * K + col] = sum;
  }
}

// Разбить матрицы на тайлы и загружать их в shared memory
__global__ void better_matrix_multiplication_kernel(const float *A,
                                                    const float *B, float *C,
                                                    int M, int N, int K) {


  // Shared memory - быстрая память внутри SM (streaming multiprocessor)
  // Доступ к ней быстрее, чем к global memory
  // Но она ограничена по размеру (обычно 48 КБ на SM), latency ~ как L1 cache
  // Используется для совместного использования данных между потоками в одном блоке
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  const int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {

    int Acol = t * TILE + threadIdx.x;
    int Brow = t * TILE + threadIdx.y;

    if (row < M && Acol < N)
      As[threadIdx.y][threadIdx.x] = A[row * N + Acol];
    else
      As[threadIdx.y][threadIdx.x] = 0.0f;

    if (Brow < N && col < K)
      Bs[threadIdx.y][threadIdx.x] = B[Brow * K + col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads(); // гарантирует, что все потоки внутри блока дошли до этого места
                     // и данные в shared memory готовы к использованию

    for (int k = 0; k < TILE; ++k)
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

    __syncthreads();
  }

  if (row < M && col < K)
    C[row * K + col] = sum;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                      int K) {
  // A (MxN), B (NxK), C (MxK)
  // dim3 - специальная структура в CUDA, которая просто хранит 3 целых числа
  // unsigned int x, y, z; по умолчанию x = 1, y = 1, z = 1 потом в потоке можно
  // использовать threadIdx.x (0 .. 15), threadIdx.y (0 .. 15), threadIdx.z (0)
  dim3 threadsPerBlock(16, 16); // один cuda-block = 16 * 16 потоков

  // grid.x покрывает столбцы C (K)
  // grid.y покрывает строки C (M)
  // каждый блок вычисляет плитку 16х16 элементов C
  dim3 blocksPerGrid(
      (K + threadsPerBlock.x - 1) /
          threadsPerBlock.x, // округление вверх (если просто K /
                             // threadsPerBloc.x, то может не хватить блоков)
      (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);
  cudaDeviceSynchronize();
}