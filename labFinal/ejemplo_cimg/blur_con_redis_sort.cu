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

// Kernel para calcular el bit n-ésimo
__global__ void compute_bit_kernel(int* d_input, int* d_bit, int bit, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_bit[idx] = (d_input[idx] >> bit) & 1;
    }
}

void radix_sort(int* d_input, int* d_output, int n, int num_bits) {
    thrust::device_vector<int> input(d_input, d_input + n);
    thrust::device_vector<int> output(n);
    thrust::device_vector<int> bit(n);
    thrust::device_vector<int> temp(n);

    for (int bit_index = 0; bit_index < num_bits; ++bit_index) {
        compute_bit_kernel<<<(n + 255) / 256, 256>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(bit.data()), bit_index, n);
        cudaDeviceSynchronize();

        // Exclusive scan on the negation of the bit array to count the falses
        thrust::exclusive_scan(thrust::make_transform_iterator(bit.begin(), thrust::logical_not<int>()), 
                               thrust::make_transform_iterator(bit.end(), thrust::logical_not<int>()), 
                               temp.begin());

        int total_falses = temp[n - 1] + !bit[n - 1];

        // Scatter the elements into the correct positions
        thrust::scatter_if(input.begin(), input.end(), temp.begin(), bit.begin(), output.begin());
        thrust::scatter_if(input.begin(), input.end(), thrust::make_counting_iterator(0), bit.begin(), output.begin() + total_falses);

        // Swap the input and output
        input.swap(output);
    }

    // Copy the result back to the output pointer
    cudaMemcpy(d_output, thrust::raw_pointer_cast(input.data()), n * sizeof(int), cudaMemcpyDeviceToDevice);
}

__global__ void filtro_mediana_kernel(float* d_input, float* d_output, int width, int height, int W, float* d_medians, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float median = d_medians[idx];

    if (fabs(d_input[idx] - median) > threshold) {
        d_output[idx] = median;
    } else {
        d_output[idx] = d_input[idx];
    }
}

// Función para calcular las medianas usando Thrust en el host
void calculate_medians(float* h_input, float* h_medians, int width, int height, int W) {
    int radius = W / 2;
    int max_neighborhood_size = W * W;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            thrust::host_vector<float> neighborhood;

            // Construir el vecindario
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        neighborhood.push_back(h_input[ny * width + nx]);
                    }
                }
            }

            // Ordenar el vecindario y encontrar la mediana
            thrust::sort(neighborhood.begin(), neighborhood.end());
            float median = neighborhood[neighborhood.size() / 2];

            h_medians[y * width + x] = median;
        }
    }
}

#define NS(f, elap)                                                                                           \
{                                                                                                            \
    struct timespec t_ini, t_fin;                                                                            \
    clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                                  \
    f;                                                                                                       \
    clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                                  \
    elap = (t_fin.tv_sec - t_ini.tv_sec) * 1000000000.0 + (t_fin.tv_nsec - t_ini.tv_nsec);                   \
}


void filtro_mediana_gpu(float* img_in, float* img_out, int width, int height, int W) {
    // Reservar memoria en la GPU
    float* d_in, *d_out, *d_medians;
    size_t size = width * height * sizeof(float);
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMalloc(&d_medians, size);

    cudaMemcpy(d_in, img_in, size, cudaMemcpyHostToDevice);

    float* h_medians = (float*)malloc(size);

    double elapsed_ns;
    NS(calculate_medians(img_in, h_medians, width, height, W), elapsed_ns);

    printf("Elapsed time: %.0f ns\n", elapsed_ns);

    cudaMemcpy(d_medians, h_medians, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    filtro_mediana_kernel<<<gridDim, blockDim>>>(d_in, d_out, width, height, W, d_medians, 0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error en el kernel: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel ejecutado correctamente\n");
    }

    cudaMemcpy(img_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_medians);
    free(h_medians);
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

#include <cuda_runtime.h>
#include <iostream>

__global__ void radix_sort_kernel(int* input, int* output, int n, int num_bits) {
    extern __shared__ int shared_data[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int* temp = &shared_data[0];
    int* scanned = &shared_data[blockDim.x];

    for (int bit = 0; bit < num_bits; bit++) {
        int mask = 1 << bit;
        
        // Step 1: Determine bit value and store in shared memory
        if (tid < n) {
            temp[tid] = (input[tid] & mask) ? 1 : 0;
        }
        __syncthreads();

        // Step 2: Perform exclusive scan (prefix sum)
        // Implementing a naive scan for demonstration purposes
        scanned[tid] = (tid > 0) ? temp[tid - 1] : 0;
        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            __syncthreads();
            int val = scanned[tid];
            if (tid >= offset) val += scanned[tid - offset];
            __syncthreads();
            scanned[tid] = val;
        }

        // Step 3: Scatter based on scanned results and bit values
        if (tid < n) {
            int pos = (temp[tid] == 1) ? scanned[tid] + __syncthreads_count(temp[tid]) : tid - scanned[tid];
            output[pos] = input[tid];
        }
        __syncthreads();

        // Swap input and output buffers
        if (tid < n) {
            input[tid] = output[tid];
        }
    }
}

int main() {
    const int n = 1024;  // Example size
    const int num_bits = sizeof(int) * 8;  // Total bits in an integer
    int* d_input, *d_output;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    // Initialize and copy data to device, execute kernel, etc.

    dim3 blocks((n + 255) / 256);
    dim3 threads(256);

    radix_sort_kernel<<<blocks, threads, 2 * threads.x * sizeof(int)>>>(d_input, d_output, n, num_bits);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
