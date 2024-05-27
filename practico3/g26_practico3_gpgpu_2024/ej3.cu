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

int* randomMatrix(int m, int n){
    int* A = (int*) malloc(m * n *sizeof(int));

    for (int i = 0; i < m * n; i++){
        A[i] = rand()%100;
    }
    return A;
}

void printMatrix(int* A, int m, int n){
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++){
            printf("%d ", A[i * n + j]);
        }
        printf("\n");
    }
}

__global__ void matrixVectorKernel(int* A, int* v, int* Av, int numRows, int numCols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows){
        int sum = 0;
        for (int col = 0; col < numCols; col++) {
            sum += A[row * numCols + col] * v[col];
        }
    Av[row] = sum;
    }
}

int main(){

    /* definir tamanios de matriz*/
    int m = 10240;
    int n = 256;

    int inputMatrixSize = m*n*sizeof(int);
    int outputSize = m*1*sizeof(int);
    int vectorSize = n*1*sizeof(int);

    /* generar matriz, vector y matriz resultado*/
    int* h_M = randomMatrix(m,n);
    int* h_v = randomMatrix(n,1);
    int* h_Mv = (int*)malloc(outputSize);

    /* reservar memoria en la GPU */
    int *d_M;
    int *d_v;
    int *d_Mv;
	CUDA_CHK(cudaMalloc((void **)&d_M, inputMatrixSize)); //matriz de entrada
    CUDA_CHK(cudaMalloc((void **)&d_v, vectorSize)); //vector
    CUDA_CHK(cudaMalloc((void **)&d_Mv, outputSize)); //matriz de salida

	/* copiar los datos de entrada a la GPU */
 	cudaMemcpy(d_M, h_M, inputMatrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, vectorSize, cudaMemcpyHostToDevice);

    /* Ej 3 */
    int threadsPerBlock = 256;
    int numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < 10; i++)
        matrixVectorKernel<<<numBlocks, threadsPerBlock>>>(d_M, d_v, d_Mv, m, n);
    
    /* Copiar los datos de salida a la CPU en h_message */
 	cudaMemcpy(h_Mv, d_Mv, outputSize, cudaMemcpyDeviceToHost);

    /* Liberar memoria */
    free(h_M);
    free(h_v);
    free(h_Mv);
    cudaFree(d_M);
    cudaFree(d_v);
    cudaFree(d_Mv);

    return 1;
 }