#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "cuda.h"
#include <cuda_runtime.h>
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
#include <iostream>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__ )

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

__global__ void cargarYExtraerVentanaKernel(float* d_input, float* ventanaGlobal, int width, int height, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = W / 2;
    int blockWidth = blockDim.x + 2 * radius;

    if (x >= width || y >= height) return;

    extern __shared__ float sharedMem[];

    // Índices locales dentro de la memoria compartida
    int localX = threadIdx.x + radius;
    int localY = threadIdx.y + radius;

    // Cargar datos en memoria compartida con verificación de límites
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int globalX = x + dx;
            int globalY = y + dy;
            int indexSharedMem = (localY + dy) * blockWidth + (localX + dx);

            if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                sharedMem[indexSharedMem] = d_input[globalY * width + globalX];
            } else {
                sharedMem[indexSharedMem] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Copiar la ventana calculada al arreglo global de salida
    int idx = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int sharedIndex = (localY + dy) * blockWidth + (localX + dx);
            ventanaGlobal[(y * width + x) * W * W + idx++] = sharedMem[sharedIndex];
        }
    }
}

__device__ void split(float* d_input, float* d_output, unsigned int bit, unsigned int d_in_len, float* s_data) {
    unsigned int thid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + thid;
    float* e = s_data;
    float* f = s_data + blockDim.x;
    unsigned int int_val;

    __syncthreads();
    if (threadIdx.x == 1 && threadIdx.y == 0) {  // Imprime la ventana para el primer píxel
        printf("Ventana original para el píxel SPIT (1,0):\n");
        for (int i = 0; i < 9; i++) {
            printf("%f ", d_input[i]);
        }
        printf("\n");
    }
    __syncthreads();

    if (global_id < d_in_len) {
        float val = d_input[global_id];
        int_val = __float_as_int(val);
        unsigned int bit_value = (int_val >> bit) & 1;
        e[thid] = 1 - bit_value;
    } else {
        e[thid] = 0;
    }
    f[thid] = e[thid];
    __syncthreads();

    // Suma prefija exclusiva
    for (unsigned int d = 1; d < blockDim.x; d *= 2) {
        int k = 2 * d * thid;
        if (k + d < blockDim.x) {
            f[k + d] += f[k];
        }
        __syncthreads();
    }

    unsigned int totalFalses = f[blockDim.x - 1];
    __syncthreads();

    if (global_id < d_in_len) {
        unsigned int bit_value = (int_val >> bit) & 1;
        unsigned int output_index = bit_value == 0 ? f[thid] : f[thid] + totalFalses;
        if (threadIdx.x == 1 && threadIdx.y == 0) {
            printf("Bit: %u, thid: %u, output_index: %u, val: %f\n", bit, thid, output_index, d_input[global_id]);
            printf("e[%u]: %f, f[%u]: %f\n", thid, e[thid], thid, f[thid]);
        }


        d_output[output_index] = d_input[global_id];
    }
}



__device__ void radix_sort(float* window, unsigned int d_in_len, float* shared_mem) {
    float* d_temp = shared_mem + d_in_len;

    for (unsigned int bit = 0; bit < sizeof(unsigned int) * 8; ++bit) {
        split(window, d_temp, bit, d_in_len, shared_mem);

        float* tmp = window;
        window = d_temp;
        d_temp = tmp;
    }

    if (window != shared_mem) {
        for (unsigned int i = 0; i < d_in_len; ++i) {
            shared_mem[i] = window[i];
        }
    }
    __syncthreads();
}

__global__ void ordenarYCalcularMedianaKernel(float* ventanaGlobal, float* d_output, int width, int height, int W, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idxBase = (y * width + x) * W * W;
    float window[25];  // Asegurando que W*W <= 25

    for (int i = 0; i < W * W; i++) {
        window[i] = ventanaGlobal[idxBase + i];
    }

    extern __shared__ float shared_mem[];

    // Inicializar la memoria compartida con ceros
    for (int i = threadIdx.x; i < W * W; i += blockDim.x) {
        shared_mem[i] = 0.0f;
    }
    __syncthreads();

    if (x == 1 && y == 0) {  // Imprime la ventana para el primer píxel
        printf("Ventana original para el píxel (1,0):\n");
        for (int i = 0; i < W * W; i++) {
            printf("%f ", window[i]);
        }
        printf("\n");

        printf("Ventana ordenada para el píxel (1,0) en shared:\n");
        for (int i = 0; i < W * W; i++) {
            printf("%f ", shared_mem[i]);
        }
        printf("\n");
    }
    __syncthreads();

    radix_sort(window, W * W, shared_mem);
    __syncthreads();

    if (x == 1 && y == 0) {  // Imprime la ventana para el primer píxel
        printf("Ventana ordenada para el píxel (1,0) en window:\n");
        for (int i = 0; i < W * W; i++) {
            printf("%f ", window[i]);
        }
        printf("\n");

        printf("Ventana ordenada para el píxel (1,0) en shared:\n");
        for (int i = 0; i < W * W; i++) {
            printf("%f ", shared_mem[i]);
        }
    }

    float median = window[W * W / 2];
    d_output[y * width + x] = median;
}


void filtro_mediana_gpu(float* img_in, float* img_out, int width, int height, int W) {
    printf("Entré\n");

    float *d_in, *d_out, *d_ventana_global;
    size_t size = width * height * sizeof(float);

    size_t ventanaGlobalSize = width * height * W * W * sizeof(float);
    float* h_ventana_global = (float*)malloc(ventanaGlobalSize);

    checkCudaErrors(cudaMalloc(&d_in, size));
    checkCudaErrors(cudaMalloc(&d_out, size));
    checkCudaErrors(cudaMalloc(&d_ventana_global, ventanaGlobalSize));

    checkCudaErrors(cudaMemcpy(d_in, img_in, size, cudaMemcpyHostToDevice));

    dim3 blockDim(4, 4);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    int radius = W / 2;
    int enhancedBlockSize = blockDim.x + 2 * radius;
    size_t sharedMemSizeExtraction = enhancedBlockSize * enhancedBlockSize * sizeof(float);

    cargarYExtraerVentanaKernel<<<gridDim, blockDim, sharedMemSizeExtraction>>>(d_in, d_ventana_global, width, height, W);
    checkCudaErrors(cudaDeviceSynchronize());

    size_t sharedMemSizeSorting = 2 * W * W * sizeof(float); // Doble para almacenar tanto d_in como d_temp

    ordenarYCalcularMedianaKernel<<<gridDim, blockDim, sharedMemSizeSorting>>>(d_ventana_global, d_out, width, height, W, 0.0f);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(img_out, d_out, size, cudaMemcpyDeviceToHost));

    // Imprimir la salida del array img_out para verificar
    printf("Salida del filtro de mediana:\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%f ", img_out[y * width + x]);
        }
        printf("\n");
    }

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_ventana_global));

    printf("Filtro mediana GPU %d terminado\n", W);
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
