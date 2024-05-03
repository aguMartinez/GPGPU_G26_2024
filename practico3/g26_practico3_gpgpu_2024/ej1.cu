/* Ejercicio 1A | Practico 3 | Grupo 26 | GPGPU 2024*/

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

int *randomMatrix(int m, int n)
{
    int *A = (int *)malloc(m * n * sizeof(int));

    for (int i = 0; i < m * n; i++)
    {
        A[i] = rand()%100;
    }
    return A;
}

void printMatrix(int *A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%d ", A[i * n + j]);
        }
        printf("\n");
    }
}

__global__ void transposeKernel(int *d_M, int *d_MTrans, int m, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    d_MTrans[x + y * n] = d_M[y + x * m];
}

int main(int argc, char* argv[]){

    if (argc != 5) {
        printf("Faltaron argumentos %d \n", argc);
        return 1;
    }

    /* definir tamanios de matriz*/
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);

    int blockX = atoi(argv[3]);
    int blockY = atoi(argv[4]);

    for(int input=0; input<5; input++){
        printf("%d ",atoi(argv[input]));
    }
    
    printf("\n");

    int size = m * n * sizeof(int);

    /* generar matrices del host */
    int *h_M = randomMatrix(m, n);
    int *h_MTrans = (int *)malloc(size);

    /* reservar memoria en la GPU */
    int *d_M;
    int *d_MTrans;
    CUDA_CHK(cudaMalloc((void **)&d_M, size));      // matriz de entrada
    CUDA_CHK(cudaMalloc((void **)&d_MTrans, size)); // matriz de salida

    /* copiar los datos de entrada a la GPU */
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);

    /* Ej 1:*/
    dim3 threadsPerBlock(blockX, blockY);
    dim3 numBlocks(m / threadsPerBlock.x,
                   n / threadsPerBlock.y);

    for(int i = 0; i < 10; i++){
        transposeKernel<<<numBlocks, threadsPerBlock>>>(d_M, d_MTrans, m, n);
    }

    /* Copiar los datos de salida a la CPU en h_message */
    cudaMemcpy(h_MTrans, d_MTrans, size, cudaMemcpyDeviceToHost);

    /* Imprimir resultados*/
    /*
    printMatrix(h_M, m, n);
    printf("-----------------------\n");
    printMatrix(h_MTrans, m, n);
    */

    /* Liberar memoria */
    free(h_M);
    free(h_MTrans);
    cudaFree(d_M);
    cudaFree(d_MTrans);

    return 1;
}
