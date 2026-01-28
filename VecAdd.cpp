#include <cuda_runtime.h>


//__global__ - означает, что функция выполняется на GPU, вызывается с CPU
// запускается в виде сетки блоков потоков
// каждый поток будет обрабатывать свой i
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    //blockIdx.x   → номер блока
    // threadIdx.x  → номер потока внутри блока
    // blockDim.x   → threadsPerBlock
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256; // размер блока, сколько потоков в одном CUDA блоке
    // 256 - кратно warp size (32), хорошо заполняет SM
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    //<<<blocksPerGrid, threadsPerBlock>>> - cuda-launch синтакс
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize(); // кернел запускатся асинхронно, 
    // без него CPU пойдет дальше, с ним - блокирует CPU пока GPU не закончит
}

//Все ок, тесты пройдены
