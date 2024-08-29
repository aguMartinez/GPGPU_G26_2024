#include "cuda.h"
#include <math.h>

#ifndef _UCHAR
#define _UCHAR
    typedef unsigned char uchar;
#endif

#ifndef MAX_SHARED
#define MAX_SHARED 49152
#endif
/* 
 * V1.1 (memoria compartida):
 * - Cada hilo carga en memoria compartida la entrada de la matriz correspondiente.
 * - Las ventanas están alojada en memoria compartida.
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

__global__ void filtro_mediana_kernel_v1_2(uchar* d_input, uchar* d_output, int width, int height, int W, float threshold) {
    extern __shared__ uchar sharedMem[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = W / 2;

    // Cargo el bloque en memoria compartida
    int sharedIndex = threadIdx.x + threadIdx.y * blockDim.x;
    
    // Coordenadas del bloque
    int blockStartX = blockIdx.x * blockDim.x;
    int blockStartY = blockIdx.y * blockDim.y;

    // Variables auxiliares del cálculo de ventana
    int idx = 0;
    int currentX;
    int currentY;

    // window en shared
    uchar* window = &sharedMem[sharedIndex * W * W];

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {

            currentX = x+dx;
            currentY = y+dy;
             
            if(currentX >= 0 && currentY >= 0 && currentX < width && currentY < height){ // Si estoy dentro de la matriz:
                window[idx++] = d_input[currentX + currentY * width];
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

void filtro_mediana_gpu_v1_2(uchar* img_in, uchar* img_out, int width, int height, int W, int blockSize) {


    uchar *d_in, *d_out;
    size_t size = width * height * sizeof(uchar);

    // Asignación de memoria en el dispositivo
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Copia de datos desde el host al dispositivo
    cudaMemcpy(d_in, img_in, size, cudaMemcpyHostToDevice);
    
    // Cálculo del tamaño de bloque (48KB = 49152)
    int blockSizeDef = (blockSize <= sqrt( MAX_SHARED / (W*W) ) ) ? blockSize : floor( sqrt( MAX_SHARED / (W*W) ) );

    dim3 blockDim(blockSizeDef, blockSizeDef);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    
    // El tamaño de shared es el tamaño del bloque + cantidad de ventas * tamaño de ventanas
    size_t sharedMemSize = (blockDim.x * blockDim.y) * (W * W) * sizeof(uchar);

    // Lanzamiento del kernel
    filtro_mediana_kernel_v1_2<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, width, height, W, 0);
    
    cudaMemcpy(img_out, d_out, size, cudaMemcpyDeviceToHost);

    // Liberación de memoria del dispositivo
    cudaFree(d_in);
    cudaFree(d_out);
}
