/* Ejercicio 3 | Practico 3 | Grupo 26 | GPGPU 2024*/

#include <stdlib.h>
#include <stdio.h>
#include "cuda.h"

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define N_COLUMNS 10

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

__global__ void optimizedMatrixVectorKernel(int* A, int* v, int* Av, int numRows, int numCols) {
    __shared__ int sdata[1];

    int row = blockIdx.x;  // Cada bloque maneja una fila completa
    int col = threadIdx.x; // Cada hilo en el bloque maneja una columna

    if (threadIdx.x == 0) {
        sdata[0] = 0;  // Inicializa el acumulador en memoria compartida
    }

    __syncthreads();

    atomicAdd(&sdata[0], A[row * numCols + col] * v[col]);

    __syncthreads();

    if (threadIdx.x == 0) {
        Av[row] = sdata[0];
    }

}


int main(){

    /* definir tamanios de matriz*/
    int m = 10;
    int n = N_COLUMNS;

    int inputMatrixSize = m*n*sizeof(int);
    int outputSize = m*1*sizeof(int);
    int vectorSize = n*1*sizeof(int);

    /* generar matriz, vector y matriz resultado*/
    int* h_M = randomMatrix(m,n);
    int* h_v = randomMatrix(n,1);
    int* h_Mv = (int*)malloc(outputSize);
    memset(h_Mv, 0, outputSize);

    /* reservar memoria en la GPU */
    int *d_M;
    int *d_v;
    int *d_Mv;
	CUDA_CHK(cudaMalloc((void **)&d_M, inputMatrixSize)); //matriz de entrada
    CUDA_CHK(cudaMalloc((void **)&d_v, vectorSize)); //vector
    CUDA_CHK(cudaMalloc((void **)&d_Mv, outputSize)); //matriz de salida

	/* copiar los datos de entrada a la GPU */
 	cudaMemcpy(d_M, h_M, inputMatrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mv, h_Mv, outputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, vectorSize, cudaMemcpyHostToDevice);
/*
    dim3 threadsPerBlock(1, n);
    dim3 numBlocks(m / threadsPerBlock.x,
                   n / threadsPerBlock.y);

*/

    dim3 threadsPerBlock(n);
    dim3 numBlocks(m);

    for (int i = 0; i < 10; i ++) {
        optimizedMatrixVectorKernel<<<numBlocks, threadsPerBlock>>>(d_M, d_v, d_Mv, m, n);
    }

    // Copiar los datos del vector de salida de la GPU al host
    cudaMemcpy(h_Mv, d_Mv, outputSize, cudaMemcpyDeviceToHost);


    printMatrix(h_M, m,n);
    printf("--------------\n");
    printMatrix(h_v, n,1);
    printf("--------------\n");
    printMatrix(h_Mv, m,1);

    /* Liberar memoria */
    free(h_M);
    free(h_v);
    free(h_Mv);
    cudaFree(d_M);
    cudaFree(d_v);
    cudaFree(d_Mv);

    return 1;
 }