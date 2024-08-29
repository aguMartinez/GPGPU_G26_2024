#include "cuda.h"
#include <math.h>

#ifndef _UCHAR
#define _UCHAR
    typedef unsigned char uchar;
#endif

/* 
 * V1.0 (memoria compartida):
 * - Cada hilo carga en memoria compartida la entrada de la matriz correspondiente.
 * - Las ventanas están alojada en variable local del kernel.
 * - Cada hilo ordena su propia ventana.
 * - El algoritmo de ordenación es Bubble Sort.
*/

__device__ __forceinline__ void bubbleSort(uchar* window, int idx) {
    for (int i = 0; i < idx - 1; i++) {
        for (int j = 0; j < idx - i - 1; j++) {
            if (window[j] > window[j + 1]) {
                uchar temp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = temp;
            }
        }
    }
}

__global__ void filtro_mediana_kernel_v1_0(uchar* d_input, uchar* d_output, int width, int height, int W, float threshold) {
    extern __shared__ uchar sharedMem[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = W / 2;

    // Cargo el bloque en memoria compartida
    sharedMem[threadIdx.x + threadIdx.y * blockDim.x] = d_input[x + y * width];
    __syncthreads();

    // Coordenadas del bloque
    int blockStartX = blockIdx.x * blockDim.x;
    int blockStartY = blockIdx.y * blockDim.y;

    // Variables auxiliares del cálculo de ventana
    int idx = 0;
    int currentX;
    int currentY;

    /* 
     * Obtengo los valores de la ventana de la siguiente manera:
     *  - Si la coordenada pertenece a la matriz y está dentro del bloque leo desde shared.
     *  - Si la coordenada pertenece a la matriz y escapa al bloque leo desde global.
     *  - Si la coordenada no pertenece a la matriz fuerzo un 0.
    */

    // TODO para 1.1: Pasar a global o shared
    uchar window[15 * 15];

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {

            currentX = x+dx;
            currentY = y+dy;
             
            if(currentX >= 0 && currentY >= 0 && currentX < width && currentY < height){ // Si estoy dentro de la matriz:
                window[idx++] = (currentX >= blockStartX) && 
                                (currentX < blockStartX + blockDim.x) && 
                                (currentY >= blockStartY) && 
                                (currentY < blockStartY + blockDim.y) 
                                ? sharedMem[(threadIdx.x+dx) + (threadIdx.y+dy) * blockDim.x] : d_input[currentX + currentY * width]; // Leo desde shared si estoy dentro del bloque, o desde global sino
            } else  // Si no estoy dentro de la matriz
                window[idx++] = 0;
            
        }
    }

    // Ordeno los valores de la ventana
    bubbleSort(window, idx);

    uchar median = window[idx / 2];

    if (abs(d_input[y * width + x] - median) > threshold) {
        d_output[y * width + x] = median;
    } else {
        d_output[y * width + x] = d_input[y * width + x];
    }
}

void filtro_mediana_gpu_v1_0(uchar* img_in, uchar* img_out, int width, int height, int W, int blockSize) {

    uchar *d_in, *d_out;
    size_t size = width * height * sizeof(uchar);
    size_t windows_size = width * height * W * W * sizeof(uchar);

    // Asignación de memoria en el dispositivo
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Copia de datos desde el host al dispositivo
    cudaMemcpy(d_in, img_in, size, cudaMemcpyHostToDevice);

    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    
    // El tamaño de shared es el tamaño del bloque
    size_t sharedMemSize = blockDim.x * blockDim.y * sizeof(uchar);

    // Lanzamiento del kernel
    filtro_mediana_kernel_v1_0<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, width, height, W, 0);
    
    cudaMemcpy(img_out, d_out, size, cudaMemcpyDeviceToHost);

    // Liberación de memoria del dispositivo
    cudaFree(d_in);
    cudaFree(d_out);
}
