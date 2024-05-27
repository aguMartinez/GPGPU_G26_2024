/* Ejercicio 2 | Practico 3 | Grupo 26 | GPGPU 2024 */

#include <stdlib.h>
#include <stdio.h>
#include "cuda.h"

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int* randomMatrix(int n, int m){
    int* A = (int*) malloc(n * m *sizeof(int));

    for (int i = 0; i < n * m; i++) {
        A[i] = rand() % 100;
    }
    return A;
}

void printMatrix(int* A, int n, int m){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", A[i * m + j]);
        }
        printf("\n");
    }
}

__global__ void sum4thPosition_kernel(int* d_M, int* d_MRes, int n, int m){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x+4 < n && y < m){
        d_MRes[x + y * n] = d_M[x + y * n] + d_M[(x+4) + y * n];
    }
}

int main(){

    /* definir tamanios de matriz*/
    int n = 256;
    int m = n;
    int size = n*m*sizeof(int);

    /* generar matriz*/
    int* h_M = randomMatrix(n,m);
    int* h_MRes = (int*)malloc(size);

    /* reservar memoria en la GPU */
    int* d_M;
    int* d_MRes;
	CUDA_CHK(cudaMalloc((void **)&d_M, size)); //matriz de entrada
    CUDA_CHK(cudaMalloc((void **)&d_MRes, size)); //matriz de salida

	/* copiar los datos de entrada a la GPU */
 	cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);

    /* Ej 2a */
	dim3 threadsPerBlock(32,32);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sum4thPosition_kernel<<<numBlocks, threadsPerBlock>>>(d_M, d_MRes, n, m);

    /* Copiar los datos de salida a la CPU */
 	cudaMemcpy(h_MRes, d_MRes, size, cudaMemcpyDeviceToHost);

    /* Imprimir resultados*/
    printMatrix(h_M, n, m);
    printf("-----------------------\n");
    printMatrix(h_MRes, n, m);

    /* Liberar memoria */
    free(h_M);
    free(h_MRes);
    cudaFree(d_M);
    cudaFree(d_MRes);

    return 1;
 }