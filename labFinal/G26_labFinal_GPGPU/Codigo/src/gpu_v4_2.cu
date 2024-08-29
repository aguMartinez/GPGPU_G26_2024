#include "cuda.h"
#include <math.h>
#include <stdio.h>

#ifndef _UCHAR
#define _UCHAR
    typedef unsigned char uchar;
#endif

/* 
 * V4.2 (Bucket sort):
 * - Cada hilo carga en memoria compartida la entrada de la matriz correspondiente.
 * - Cada hilo ordena su propia ventana.
 * - Cambia el cálculo de mediana:
 * - Se utiliza Bucket Sort para contar las ocurrencias de valores y se calcula la mediana a partir de ellos sin ordenar el arreglo.
 *   4.2: el arreglo de buckets pasa a memoria local 
*/

__global__ void filtro_mediana_kernel_v4_2(uchar* d_input, uchar* d_output, int width, int height, int W, float threshold) {

    extern __shared__ uchar sharedMem[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = W / 2;

    // Cargo el bloque en memoria compartida
    int sharedIndex = threadIdx.x + threadIdx.y * blockDim.x;

    sharedMem[sharedIndex] = d_input[x + y * width];
    __syncthreads();
    
    // Coordenadas del bloque
    int blockStartX = blockIdx.x * blockDim.x;
    int blockStartY = blockIdx.y * blockDim.y;

    // Variables auxiliares del cálculo de ventana
    int idx = 0;
    int currentX;
    int currentY;

    // buckets en local
    uchar buckets[256] = {0}; // inicializa en 0, magia de P2

    // En lugar de usar ventana se llenan los buckets directamente
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {

            currentX = x + dx;
            currentY = y + dy;
             
            if(currentX >= 0 && currentY >= 0 && currentX < width && currentY < height) // Si estoy dentro de la matriz:
                if((currentX >= blockStartX) && (currentX < blockStartX + blockDim.x) &&  // Si estoy dentro del bloque en shared:
                   (currentY >= blockStartY) && (currentY < blockStartY + blockDim.y)) 
                    
                    buckets[ sharedMem[(threadIdx.x+dx) + (threadIdx.y+dy) * blockDim.x] ]++;
                else
                    buckets[ d_input[currentX + currentY * width] ]++;
             else  // Si no estoy dentro de la matriz
                buckets[0]++;
        }
    }

    // Ordeno los valores de la ventana
    int cont = 0;
    int i = 0;

    // Camino hasta la mitad del arreglo
    for (i = 0; (i < 256) && (cont <= W*W/2); i++)
        cont += buckets[i];

    d_output[y * width + x] = i-1;
}

void filtro_mediana_gpu_v4_2(uchar* img_in, uchar* img_out, int width, int height, int W, int blockSize) {
    uchar *d_in, *d_out;
    size_t size = width * height * sizeof(uchar);

    cudaError_t err;

    // Asignación de memoria en el dispositivo
    cudaMalloc(&d_in, size);
    
    cudaMalloc(&d_out, size);

    // Copia de datos desde el host al dispositivo
    cudaMemcpy(d_in, img_in, size, cudaMemcpyHostToDevice);

    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    
    // Tamaño de memoria compartida
    size_t sharedMemSize = (blockDim.x * blockDim.y) * (1 + 256) * sizeof(uchar);

    // Lanzamiento del kernel
    filtro_mediana_kernel_v4_2<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, width, height, W, 0);

    // Copia de datos desde el dispositivo al host
    cudaMemcpy(img_out, d_out, size, cudaMemcpyDeviceToHost);

    // Liberación de memoria del dispositivo
    cudaFree(d_in);
    cudaFree(d_out);
}
