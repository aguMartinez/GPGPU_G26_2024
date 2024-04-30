/* Ejercicio 3 | Practico 3 | Grupo 26 | GPGPU 2024*/

#include <stdlib.h>
#include <stdio.h>
#include "cuda.h"

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define N_COLUMNS 10
#define N_ROWS 10

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

/*
__global__ void matrixElementMultKernel(int* A, int* v, int numRows, int numCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < numRows && col < numCols) {
        A[row * numCols + col] = A[row * numCols + col] * v[col];
    }
}
*/

__global__ void matrixVectorKernel(int* A, int* v, int* Av, int numRows, int numCols){
    __shared__ int sdata[N_COLUMNS];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    atomicAdd(&sdata[x], A[x * numCols + y] * v[y] + sdata[x]);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        Av[x] += sdata[x];
    }
}

/*
    int i = blockDim.x / 2;
    while (i != 1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
        i /= 2;
    }
*/
__global__ void treeRowReductionKernel(int *input, int *output, int numCols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ int sdata[];
    int i = row * numCols + tid;
    sdata[tid] = input[i];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[row] = sdata[0];
}

__global__ void matrixVectorKernelOptimized(int* A, int* v, int* Av, int numRows, int numCols) {
    extern __shared__ int s_v[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (threadIdx.x < numCols) {
        s_v[threadIdx.x] = v[threadIdx.x];
    }
    
    __syncthreads();

    if (row < numRows) {
        int sum = 0;
        for (int col = 0; col < numCols; col++) {
            sum += A[row * numCols + col] * s_v[col];
        }
        Av[row] = sum;
    }
}



int main(){

    /* definir tamanios de matriz*/
    int m = 10;
    int n = 10;

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
/*

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + 15) / 16, (n + 15) / 16);
    int threadsPerBlock1 = m; 
    int numBlocks1 = m;
*/
    //size_t sharedMemorySize = n * sizeof(int);

    dim3 threadsPerBlock(2, 2);
    dim3 numBlocks(m / threadsPerBlock.x,
                   n / threadsPerBlock.y);

    for (int i = 0; i < 10; i ++) {
        /* Ej 3 */
        matrixVectorKernel<<<numBlocks, threadsPerBlock>>>(d_M, d_v, d_Mv, m, n);
        //matrixElementMultKernel<<<numBlocks, threadsPerBlock>>>(d_M, d_v, m, n);

        //int threadsPerBlockRed = 256;
        //int numBlocksRed = (m * n + threadsPerBlockRed - 1) / threadsPerBlockRed;
        //reduction<<<numBlocksRed, threadsPerBlockRed, threadsPerBlockRed * sizeof(int)>>>(d_Mv, d_M);
    
        //treeRowReductionKernel<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_M, d_Mv, n);
        
    }

/*
    for (int i = 0; i < 10; i++) {
        matrixVectorKernelOptimized<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_M, d_v, d_Mv, m, n);
        cudaDeviceSynchronize(); // Sincronizar después de cada ejecución del kernel
    }
*/
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