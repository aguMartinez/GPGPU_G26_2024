#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nvToolsExt.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

__global__ void filtro_mediana_kernel(float* d_input, float* d_output, int width, int height, int W, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = W / 2;
    int max_neighborhood_size = W * W;
    float* neighborhood = (float*)malloc(max_neighborhood_size * sizeof(float));
    int idx = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                neighborhood[idx++] = d_input[ny * width + nx];
            }
        }
    }

    for (int i = 0; i < idx - 1; i++) {
        for (int j = 0; j < idx - i - 1; j++) {
            if (neighborhood[j] > neighborhood[j + 1]) {
                float temp = neighborhood[j];
                neighborhood[j] = neighborhood[j + 1];
                neighborhood[j + 1] = temp;
            }
        }
    }

    float median = neighborhood[idx / 2];

    if (fabs(d_input[y * width + x] - median) > threshold) {
        d_output[y * width + x] = median;
    } else {
        d_output[y * width + x] = d_input[y * width + x];
    }
}


void filtro_mediana_gpu(float* img_in, float* img_out, int width, int height, int W) {
    float* d_in, *d_out;
    size_t size = width * height * sizeof(float);
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, img_in, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    filtro_mediana_kernel<<<gridDim, blockDim>>>(d_in, d_out, width, height, W, 0);

    cudaMemcpy(img_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}
void filtro_mediana_cpu(float * img_in, float * img_out, int width, int height, int W) {
    int radius = W / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::vector<float> neighborhood;

            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        neighborhood.push_back(img_in[ny * width + nx]);
                    }
                }
            }

            std::nth_element(neighborhood.begin(), neighborhood.begin() + neighborhood.size() / 2, neighborhood.end());
            float median = neighborhood[neighborhood.size() / 2];

            int threshold = 0;
            // printf("Median: %f\n", fabs(img_in[y * width + x] - median));
            
            if (fabs(img_in[y * width + x] - median) > threshold) {
                img_out[y * width + x] = median;
            } else {
                img_out[y * width + x] = img_in[y * width + x];
            }
            
        }
    }
}
