#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = width * height * 4; // RGBA
    if (idx < total_pixels) {
            if (idx % 4 != 3) // не инвертировать альфа канал
                image[idx] = 255 - image[idx];
    }
}

__global__ void better_invert_kernel(unsigned char* image, int width, int height) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (pixel < total_pixels) {
        int i = pixel * 4;
        image[i + 0] = 255 - image[i + 0]; // R
        image[i + 1] = 255 - image[i + 1]; // G
        image[i + 2] = 255 - image[i + 2]; // B
        // image[i + 3] untouched (A)
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)


extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
