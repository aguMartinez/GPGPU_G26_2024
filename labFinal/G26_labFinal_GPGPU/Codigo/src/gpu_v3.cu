#include <cuda_runtime.h>
#include <math.h>

#ifndef _UCHAR
#define _UCHAR
    typedef unsigned char uchar;
#endif

#ifndef MAX_SHARED
#define MAX_SHARED 49152
#endif

/*
 * V3.0 (Radix sort):
 * - Cada hilo carga en memoria compartida la entrada de la matriz correspondiente.
 * - Cada hilo ordena su propia ventana.
 * - Cambia el cálculo de mediana:
 * - Se utiliza Radix Sort
*/

__device__ __forceinline__ 
void exclusive_scan(uchar * temp, int num_elements) {
    int previous = 0;
    for (int i = 0; i < num_elements; i++) {
        int current = temp[i];
        temp[i] = previous;
        previous += current;
    }
}

__device__ __forceinline__ 
void split(uchar* input, uchar* output, uchar* temp, int window_size, int bit) {
    int mask = 1 << bit;
    int int_representation;
    int bit_value;

    for (int i = 0; i < window_size; i++) {
        int_representation = static_cast<int>(input[i]);
        bit_value = (int_representation & mask) >> bit;
        temp[i] = (bit_value == 0) ? 1 : 0;
    }

    int ult_valor_temp = temp[window_size - 1];

    exclusive_scan(temp, window_size);

    int totalFalses = temp[window_size - 1] + ult_valor_temp;
    
    int index;

    for (int i = 0; i < window_size; i++) {
        index = temp[i];
        if ((static_cast<int>(input[i]) & mask) >> bit) {
            index = i - temp[i] + totalFalses;
        } else {
            index = temp[i];
        }

        output[index] = input[i];
    }

    for (int i = 0; i < window_size; i++) {
        input[i] = output[i];
    }

}

__global__
void filtro_mediana_kernel_v3(uchar* d_input, uchar* img_out, int width, int height, int W) {

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

    /* 
     * Obtengo los valores de la ventana de la siguiente manera:
     *  - Si la coordenada pertenece a la matriz y está dentro del bloque leo desde shared.
     *  - Si la coordenada pertenece a la matriz y escapa al bloque leo desde global.
     *  - Si la coordenada no pertenece a la matriz fuerzo un 0.
    */

    // Variables auxiliares del cálculo de ventana
    int idx = 0;
    int currentX;
    int currentY;

    // window en shared
    uchar* window = &sharedMem[blockDim.x * blockDim.y + sharedIndex * 3 * W * W];
    uchar* temp = &window[W * W];
    uchar* sorted_window = &temp[W * W];

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {

            currentX = x+dx;
            currentY = y+dy;
             
            if(currentX >= 0 && currentY >= 0 && currentX < width && currentY < height){
                window[idx++] = (currentX >= blockStartX) && 
                                (currentX < blockStartX + blockDim.x) && 
                                (currentY >= blockStartY) && 
                                (currentY < blockStartY + blockDim.y) 
                                ? sharedMem[(threadIdx.x+dx) + (threadIdx.y+dy) * blockDim.x] : d_input[currentX + currentY * width];
            } else
                window[idx++] = 0;
        }
    }


    // Ordeno los valores de la ventana
    char window_size = W * W;
    for (int bit = 0; bit < 8; bit++) {
        split(window, sorted_window, temp, window_size, bit);
    }

    img_out[y * width + x] = sorted_window[W * W / 2];
}


void filtro_mediana_gpu_v3(uchar* img_in, uchar* img_out, int width, int height, int W, int blockSize) {
    uchar *d_in, *d_out;
    size_t size = width * height * sizeof(uchar);

    cudaMalloc(&d_in, size);
    
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, img_in, size, cudaMemcpyHostToDevice);
    
    // Cálculo del tamaño de bloque
    int blockSizeDef = (blockSize <= sqrt( MAX_SHARED / (1 + W * W * 3) ) ) ? blockSize : floor( sqrt( MAX_SHARED / (1 + W * W * 3) ) );

    dim3 blockDim(blockSizeDef, blockSizeDef);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    size_t sharedMemSize = (blockDim.x * blockDim.y) * (1 + W * W * 3) * sizeof(uchar);

    filtro_mediana_kernel_v3<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, width, height, W);

    cudaMemcpy(img_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}
