/* Ejercicio 1 | Practico 4 | Grupo 26 | GPGPU 2024*/

#include <stdlib.h>
#include <stdio.h>
#include "cuda.h"

#define CUDA_CHK(ans)                         \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

int *randomMatrix(int f, int c)
{
    int *A = (int *)malloc(f * c * sizeof(int));

    for (int i = 0; i < f * c; i++)
    {
        A[i] = i;
    }
    return A;
}

int matrixIsTransposed(int * m1, int * m2, int c, int f) {

    for (int i = 0; i < f; i++) {
        for (int j = 0; j < c; j ++) {
            if (m1[i * c + j] != m2[j*f + i]) {
                return 0;
            }
        }
    }
    return 1;
}

void printMatrix(int *A, int f, int c)
{
    for (int i = 0; i < f; i++)
    {
        for (int j = 0; j < c; j++)
        {
            printf("%d ", A[i * c + j]);
        }
        printf("\n");
    }
}

__global__ void transposeKernel(int *d_M, int *d_MTrans, int f, int c){
    extern __shared__ int tile[];

    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    

    tile[threadIdx.x * blockDim.y + threadIdx.y] = d_M[(global_y) * c + (blockIdx.x * blockDim.x + threadIdx.x)];

    __syncthreads();

    int num_pos = threadIdx.y * blockDim.x + threadIdx.x;

    int offset = blockIdx.x * blockDim.x * f + blockIdx.y * blockDim.y;

    int posTile = (num_pos / blockDim.y * f) + (num_pos % blockDim.y);

    d_MTrans[offset + posTile] = tile[num_pos];
}

__global__ void transposeKernelDummy(int *d_M, int *d_MTrans, int f, int c){
    extern __shared__ int tile[];

    tile[threadIdx.x * (blockDim.y+1) + threadIdx.y] = d_M[(blockIdx.y * blockDim.y + threadIdx.y) * c + (blockIdx.x * blockDim.x + threadIdx.x)];

    __syncthreads();

    //num_pos identifica al n-esimo thread del bloque
    int num_pos_tile = threadIdx.y * blockDim.x + threadIdx.x + threadIdx.y;
    int num_pos = threadIdx.y * blockDim.x + threadIdx.x;

    //offset es la posicion del elemento (0,0) del  bloque a trapsoner en la matriz resultado
    int offset = blockIdx.x * blockDim.x * f + blockIdx.y * blockDim.y;
 
    int posTile = (num_pos / blockDim.y * f) + (num_pos % blockDim.y);

    int pos_dM = offset + posTile;


    d_MTrans[pos_dM] = tile[num_pos_tile];
}


int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        printf("Faltaron argumentos %d \n", argc);
        return 1;
    }

    /* definir tamanios de matriz*/
    int f = atoi(argv[1]);
    int c = atoi(argv[2]);

    int blockX = atoi(argv[3]);
    int blockY = atoi(argv[4]);

    for (int input = 1; input < 5; input++)
    {
        printf("%d ", atoi(argv[input]));
    }

    printf("\n");

    int size = f * c * sizeof(int);

    /* generar matrices del host */
    int* h_M = randomMatrix(f, c);
    int* h_MTrans = (int *)malloc(size);

    /* reservar memoria en la GPU */
    int *d_M;
    int *d_MTrans;
    CUDA_CHK(cudaMalloc((void **)&d_M, size));      // matriz de entrada
    CUDA_CHK(cudaMalloc((void **)&d_MTrans, size)); // matriz de salida

    /* copiar los datos de entrada a la GPU */
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);

    /* Ej 1a:*/
    dim3 threadsPerBlock(blockX, blockY);
    dim3 numBlocks((c + blockX - 1) / blockX, (f + blockY - 1) / blockY);

    int sizeTile = blockX * blockY * sizeof(int);

    for (int i = 0; i < 10; i++)
    {
        transposeKernel<<<numBlocks, threadsPerBlock, sizeTile>>>(d_M, d_MTrans, f, c);
    }

    /* Copiar los datos de salida a la CPU en h_message */
    cudaMemcpy(h_MTrans, d_MTrans, size, cudaMemcpyDeviceToHost);

    /* Imprimir resultados*/

    printf("transpuesta: %d \n",matrixIsTransposed(h_M,h_MTrans,c,f));

    free(h_MTrans);
    cudaFree(d_MTrans);

    /*Ej 1b:*/

    h_MTrans = (int *)malloc(size);
    CUDA_CHK(cudaMalloc((void **)&d_MTrans, size));


    int sizeTileDummy = (blockX) * (blockY+1) * sizeof(int);

    for (int i = 0; i < 10; i++)
    {
        transposeKernelDummy<<<numBlocks, threadsPerBlock, sizeTileDummy>>>(d_M, d_MTrans, f, c);
    }

    /* Copiar los datos de salida a la CPU en h_message */
    cudaMemcpy(h_MTrans, d_MTrans, size, cudaMemcpyDeviceToHost);
    


    /* Imprimir resultados*/
    printf("transpuesta Dummy: %d ",matrixIsTransposed(h_M,h_MTrans,c,f));
    

    /* Liberar memoria */
    free(h_M);
    free(h_MTrans);
    cudaFree(d_M);
    cudaFree(d_MTrans);

    return 1;
}
