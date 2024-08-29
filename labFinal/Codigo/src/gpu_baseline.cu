#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nvToolsExt.h"

#ifndef _UCHAR
#define _UCHAR
    typedef unsigned char uchar;
#endif


__global__ void filtro_mediana_kernel_baseline(uchar* d_input, uchar* d_output, uchar* d_windows, int width, int height, int W, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = W / 2;
    int max_window_size = W * W;

    int thread_id = y * width + x;

    uchar* window = &d_windows[thread_id * max_window_size];

    int idx = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                window[idx++] = d_input[ny * width + nx];
            } else {
                window[idx++] = 0.0f;
            }
        }
    }

    for (int i = 0; i < idx - 1; i++) {
        for (int j = 0; j < idx - i - 1; j++) {
            if (window[j] > window[j + 1]) {
                uchar temp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = temp;
            }
        }
    }

    uchar median = window[idx / 2];

    if (abs(d_input[y * width + x] - median) > threshold) {
        d_output[y * width + x] = median;
    } else {
        d_output[y * width + x] = d_input[y * width + x];
    }
}

void filtro_mediana_gpu_baseline(uchar* img_in, uchar* img_out, int width, int height, int W, int blockSize) {
    uchar* d_in, *d_out, *d_windows;
    size_t size = width * height * sizeof(uchar);
    size_t windows_size = width * height * W * W * sizeof(uchar);
    
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMalloc(&d_windows, windows_size);

    cudaMemcpy(d_in, img_in, size, cudaMemcpyHostToDevice);

    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    filtro_mediana_kernel_baseline<<<gridDim, blockDim>>>(d_in, d_out, d_windows, width, height, W, 0);

    cudaMemcpy(img_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_windows);
}