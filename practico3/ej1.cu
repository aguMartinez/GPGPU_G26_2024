/* Ejercicio 1 | Practico 3 | Grupo 26 | GPGPU 2024*/

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

__global__ void transpose_kernel(int* d_M, int* d_MTrans, int n){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y

    if (x < n && y < n){
        d_MTrans[y + x * n] = d_M[x + y * n];
    }
}


 int main(){

    /* definir tamanios de matriz*/
    int n = 256;
    int m = n;
    int size = n*m*sizeof(int)
    
    int* R;

    /* generar matriz*/
    int* h_M;
    h_M = randomMatrix(n,m);

    /* reservar memoria en la GPU */
    int* d_M;
    int* d_MTrans;
	CUDA_CHK(cudaMalloc((void **)&d_M, size)); //matriz de entrada
    CUDA_CHK(cudaMalloc((void **)&d_MTrans, size)); //matriz de salida

	/* copiar los datos de entrada a la GPU */

 	cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);

    /* Ej 1:*/
	int threads_per_block = 32*32;
    int blocks_per_grid = 128; //Todo: arreglar este num.

    transpose_kernel<<<blocks_per_grid, threads_per_block>>>(d_M, d_MTrans size);
    
    /* Copiar los datos de salida a la CPU en h_message */
 	cudaMemcpy(h_MTrans, d_MTrans, size, cudaMemcpyDeviceToHost);
    return 1;
 }