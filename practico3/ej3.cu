/* Ejercicio 3 | Practico 3 | Grupo 26 | GPGPU 2024*/

#include <stdlib.h>
#include <stdio.h>
#include "cuda.h"

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int* randomMatrix(int n, int m){
    int* A = (int*) malloc(n * m *sizeof(int*));

    for (int i = 0; i < n * m; i++) {
        A[i] = rand();
    }
    return A;
}

__global__ void matrixVectorKernel(int* A, int* v, int* res, int numRows, int numCols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows){
        int sum = 0;
        for (int col = 0; col < numCols; col++) {
            sum += A[row * numCols + col] * v[col];
        }
    res[row] = sum;
    }
}


 int main(){

    /* definir tamanios de matriz*/
    int n = 10240;
    int m = 256;
    int size = n*m*sizeof(int)
    
    int* R;

    /* generar matriz*/
    int* h_M;
    h_M = randomMatrix(n,m);

    /* reservar memoria en la GPU */
	CUDA_CHK(cudaMalloc((void **)&d_M, size)); //matriz de entrada
    CUDA_CHK(cudaMalloc((void **)&d_MTrans, size)); //matriz de salida

	/* copiar los datos de entrada a la GPU */
    int* d_M;
 	cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);

    /* Ej 3:*/
	int threads_per_block = 32*32;
    int blocks_per_grid = 128; //Todo: arreglar este num.

    transpose_kernel<<<blocks_per_grid, threads_per_block>>>(d_M, size);
    
    /* Copiar los datos de salida a la CPU en h_message */
 	cudaMemcpy(h_MTrans, d_MTrans, size, cudaMemcpyDeviceToHost);
    return 1;
 }