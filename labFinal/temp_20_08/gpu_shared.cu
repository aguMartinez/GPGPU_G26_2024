#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nvToolsExt.h"

__global__ void filtro_mediana_kernel(float* d_input, float* d_output, int width, int height, int W, float threshold) {
    extern __shared__ float sharedMem[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = W / 2;

    int blockWidth = blockDim.x + 2 * radius;
    int blockHeight = blockDim.y + 2 * radius;

    int localX = threadIdx.x + radius;
    int localY = threadIdx.y + radius;

    // Cargo la ventana en memoria compartida
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);

            sharedMem[(localY + dy) * blockWidth + (localX + dx)] = d_input[ny * width + nx];
        }
    }

    __syncthreads();

    // Extraigo los valores de la ventana
    float window[15 * 15]; // Usa un tamaño estático suficiente para la ventana más grande posible
    int idx = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            window[idx++] = sharedMem[(localY + dy) * blockWidth + (localX + dx)];
        }
    }

    // Ordeno los valores de la ventana
    for (int i = 0; i < idx - 1; i++) {
        for (int j = 0; j < idx - i - 1; j++) {
            if (window[j] > window[j + 1]) {
                float temp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = temp;
            }
        }
    }

    float median = window[idx / 2];

    if (fabs(d_input[y * width + x] - median) > threshold) {
        d_output[y * width + x] = median;
    } else {
        d_output[y * width + x] = d_input[y * width + x];
    }
}

void filtro_mediana_gpu(float* img_in, float* img_out, int width, int height, int W) {
    printf("Filtro mediana GPU %d \n", W);
    float* d_in, *d_out;
    size_t size = width * height * sizeof(float);
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, img_in, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    // TODO: Revisar
    size_t sharedMemSize = (blockDim.x + W - 1) * (blockDim.y + W - 1) * sizeof(float);

    filtro_mediana_kernel<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, width, height, W, 0);

    cudaDeviceSynchronize();

    cudaMemcpy(img_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    printf("Filtro mediana GPU %d terminado\n", W);
}
