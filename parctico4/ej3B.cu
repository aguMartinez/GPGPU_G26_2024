/* Ejercicio 3 | Practico 4 | Grupo 26 | GPGPU 2024 */

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define HISTOGRAM_LENGTH 8
#define FILAS 16
#define COLUMNAS 16

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int *randomMatrix(int f, int c)
{
    int *A = (int *)malloc(f * c * sizeof(int));

    for (int i = 0; i < f * c; i++)
    {
        A[i] = rand() % HISTOGRAM_LENGTH;
    }
    return A;
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

__global__ void decrypt_kernel_ej3B(int *d_M, int * d_H, int * d_MH)
{
	 __shared__ int shared_histogram[HISTOGRAM_LENGTH];
	
	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int num_pos = threadIdx.y * blockDim.x + threadIdx.x;

	if (num_pos < HISTOGRAM_LENGTH) {
		shared_histogram[num_pos] = 0;
	}

	__syncthreads();

	int elem = d_M[y * COLUMNAS + x];
	//shared_histogram[elem]++;
	atomicAdd(&shared_histogram[elem], 1);
	
	__syncthreads();

	int block_pos = blockIdx.y * gridDim.x + blockIdx.x;

	if (num_pos < HISTOGRAM_LENGTH) {
		d_MH[block_pos * HISTOGRAM_LENGTH + num_pos] = shared_histogram[num_pos];
	}
}

__global__ void reduction(int* d_MH, int numRows, int numCols, int salto) {
	extern __shared__ int intermedio[];

	int blockDimX = blockDim.x;
	
	int x = blockIdx.x * blockDimX + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int num_pos = threadIdx.y * blockDimX + threadIdx.x;

	intermedio[num_pos] = d_MH[(y+salto) * HISTOGRAM_LENGTH + x];

	__syncthreads();

	int i = blockDimX * blockDim.y / 2;
	while (i >= blockDimX) {
		if (num_pos < i) {
			//printf("(%d,%d) (%d,%d) %d + %d \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, intermedio[num_pos], intermedio[num_pos+i]);
			intermedio[num_pos] = intermedio[num_pos] + intermedio[num_pos+i];
		}
		__syncthreads();
		i = i / 2;
	}
    __syncthreads();

	// Guardar los resultado
	if (num_pos < blockDimX) {
		d_MH[y *  HISTOGRAM_LENGTH + x] = intermedio[num_pos];
	}
}

int main(int argc, char *argv[])
{

	if (argc != 3)
    {
        printf("Faltaron argumentos %d \n", argc);
        return 1;
    }

    /* definir tamanios de matriz*/
    int blockX = atoi(argv[1]);
    int blockY = atoi(argv[2]);

    for (int input = 1; input < 3; input++)
    {
        printf("%d ", atoi(argv[input]));
    }

    printf("\n");

	int sizeH = HISTOGRAM_LENGTH * sizeof(int);
	int sizeM = FILAS * COLUMNAS * sizeof(int);

	int* h_M = randomMatrix(FILAS, COLUMNAS);
	int* h_H = (int*) malloc(sizeH);

	memset(h_H, 0, sizeH);

	int* d_M; 
	int* d_H;

	/* reservar memoria en la GPU */
	CUDA_CHK(cudaMalloc((void **)&d_H, sizeH));
	CUDA_CHK(cudaMalloc((void **)&d_M, sizeM));

	/* copiar los datos de entrada a la GPU */

 	cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);
 	cudaMemcpy(d_H, h_H, sizeH, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(blockX, blockY);
	dim3 numBlocks( (COLUMNAS + blockX - 1) / blockX, (FILAS + blockY - 1) / blockY);

	int sizeMH = numBlocks.x * numBlocks.y * HISTOGRAM_LENGTH * sizeof(int);

	int* h_MH = (int*) malloc(sizeMH);
	int* d_MH;

	CUDA_CHK(cudaMalloc((void **)&d_MH, sizeMH));

    decrypt_kernel_ej3B<<<numBlocks, threadsPerBlock>>>(d_M, d_H, d_MH);

	/* Copiar los datos de salida a la CPU en h_message */
	// cudaMemcpy(h_H, d_H, sizeH, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_MH, d_MH, sizeMH, cudaMemcpyDeviceToHost);

	printf("-----------MATRIZ ORIGINAL------------\n");
	printMatrix(h_M, FILAS, COLUMNAS);

	int remainingRows = numBlocks.x * numBlocks.y;
	cudaMemcpy(h_MH, d_MH, sizeMH, cudaMemcpyDeviceToHost);
	printf("-----------MATRIZ HISTOGRAMA ANTES DEL REDUCE------------\n");
	printMatrix(h_MH, remainingRows, HISTOGRAM_LENGTH);

	
	//begin initial_reduction
	remainingRows = numBlocks.x * numBlocks.y;  //Todas las filas restantes
	blockX = 4; //Pasar por input
	blockY = 4; //Pasar por input

	dim3 threadsPerBlock2(blockX, blockY);
	dim3 numBlocks2((HISTOGRAM_LENGTH + blockX - 1) / blockX, (remainingRows + blockY - 1) / blockY);

	int sizeShared = blockX * blockY * sizeof(int);
	reduction<<<numBlocks2, threadsPerBlock2, sizeShared>>>(d_MH, remainingRows, HISTOGRAM_LENGTH, 0);
	//end initial_reduction
	/*
	//begin loop
	remainingRows = numBlocks.x;
	blockY = 4; //Idealmente se pasa por input
	blockX = 1024/blockY;
	sizeShared = blockX * blockY * sizeof(int);
	dim3 threadsPerBlock3(blockX, blockY);
	dim3 numBlocks3((HISTOGRAM_LENGTH + blockX - 1) / blockX, (remainingRows + blockY - 1) / blockY);

	reduction<<<numBlocks2, threadsPerBlock2, sizeShared>>>(d_MH, remainingRows, HISTOGRAM_LENGTH, remainingRows);

	//end loop when remainingRows = 0
	*/
	

 	cudaMemcpy(h_MH, d_MH, sizeMH, cudaMemcpyDeviceToHost);

    printf("-----------MATRIZ HISTOGRAMA LUEGO DEL REDUCE------------\n");
	printMatrix(h_MH, remainingRows, HISTOGRAM_LENGTH);

	// libero la memoria en la CPU
	free(h_M);
	free(h_H);
	free(h_MH);

	// libero la memoria en la GPU
	cudaFree(d_M);
    cudaFree(d_H);
	cudaFree(d_MH);

	return 0;
}
