/* Ejercicio 3 | Practico 4 | Grupo 26 | GPGPU 2024 */

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <math.h>

#define HISTOGRAM_LENGTH 256
#define FILAS 2160
#define COLUMNAS 3840

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

/*Imprime filas salteadas de la matriz A*/
void printMatrixSalto(int* A, int f, int c, int salto) {
    for (int i = 0; i < f; i += salto) {
        for (int j = 0; j < c; j++) {
            printf("%d ", A[i * c + j]);
        }
        printf("\n");
    }
}

void printMatrix(int* A, int f, int c)
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

__global__ void decrypt_kernel_ej3B(int *d_M, int * d_MH)
{
	__shared__ int shared_histogram[HISTOGRAM_LENGTH];
	
	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int num_pos = threadIdx.y * blockDim.x + threadIdx.x;

	if (num_pos < HISTOGRAM_LENGTH) {
		shared_histogram[num_pos] = 0;
	}

	__syncthreads();

    if (x < COLUMNAS && y < FILAS) {
        int elem = d_M[y * COLUMNAS + x];
        atomicAdd(&shared_histogram[elem], 1);
    }

	__syncthreads();

	int block_pos = blockIdx.y * gridDim.x + blockIdx.x;

	if (num_pos < HISTOGRAM_LENGTH) {
		d_MH[block_pos * HISTOGRAM_LENGTH + num_pos] = shared_histogram[num_pos];
	}
}

__global__ void reduction(int* d_MH, int numRows, int salto) {
	extern __shared__ int intermedio[];

	int blockDimX = blockDim.x;
	
	int x = blockIdx.x * blockDimX + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int num_pos = threadIdx.y * blockDimX + threadIdx.x;

	if ((y * HISTOGRAM_LENGTH + x + y * salto * HISTOGRAM_LENGTH) < (numRows * HISTOGRAM_LENGTH)) {
		intermedio[num_pos] = d_MH[y * HISTOGRAM_LENGTH + x + y * salto * HISTOGRAM_LENGTH];
	} else {
		intermedio[num_pos] = 0;
	}

	__syncthreads();

	int i = blockDimX * blockDim.y / 2;
	while (i >= blockDimX) {
		if (num_pos < i) {
			intermedio[num_pos] = intermedio[num_pos] + intermedio[num_pos+i];
		}
		__syncthreads();
		i = i / 2;
	}
    __syncthreads();

	// Guardar los resultado
	if (num_pos < blockDimX && ((y * HISTOGRAM_LENGTH + x + y * salto * HISTOGRAM_LENGTH) < (numRows * HISTOGRAM_LENGTH))) {
		d_MH[y * HISTOGRAM_LENGTH + x + y * salto * HISTOGRAM_LENGTH] = intermedio[num_pos];
	}
}

int main(int argc, char *argv[])
{
	int sharedMemPerBlock;
	cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
	printf("Max shared memory per block: %d bytes\n", sharedMemPerBlock);


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

	int sizeM = FILAS * COLUMNAS * sizeof(int);

	int* h_M = randomMatrix(FILAS, COLUMNAS);
	int* d_M; 

	CUDA_CHK(cudaMalloc((void **)&d_M, sizeM));

 	cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(blockX, blockY);
	dim3 numBlocks( (COLUMNAS + blockX - 1) / blockX, (FILAS + blockY - 1) / blockY);

	int rowsMH = numBlocks.x * numBlocks.y;
	int sizeMH = rowsMH * HISTOGRAM_LENGTH * sizeof(int);
	int remainingRows = rowsMH;

	int* h_MH = (int*) malloc(sizeMH);
	int* d_MH;


	CUDA_CHK(cudaMalloc((void **)&d_MH, sizeMH));
	printf("numblocks.x: %d, numblocks.y: %d \n", numBlocks.x, numBlocks.y);

    decrypt_kernel_ej3B<<<numBlocks, threadsPerBlock>>>(d_M, d_MH);
	
	/* Copiar los datos de salida a la CPU en h_message */
	cudaMemcpy(h_MH, d_MH, sizeMH, cudaMemcpyDeviceToHost);

	numBlocks.x = (HISTOGRAM_LENGTH + blockX - 1) / blockX;
	numBlocks.y = (remainingRows + blockY - 1) / blockY;

	int sizeShared = blockX * blockY * sizeof(int);
	long salto = 1;

	while (remainingRows > 1) {

		reduction<<<numBlocks, threadsPerBlock, sizeShared>>>(d_MH, rowsMH, salto - 1);
        cudaDeviceSynchronize();
        CUDA_CHK(cudaGetLastError());

		salto = salto * blockY;

		remainingRows = (remainingRows + blockY - 1) / blockY;

        numBlocks.y = remainingRows;
	}

 	cudaMemcpy(h_MH, d_MH, sizeMH, cudaMemcpyDeviceToHost);

	printf("-----------MATRIZ HISTOGRAMA LUEGO DEL REDUCE------------\n");
	printMatrix(h_MH, 1, HISTOGRAM_LENGTH);
	printf("---------------------------------\n");


	// libero la memoria en la CPU
	free(h_M);
	free(h_MH);

	// libero la memoria en la GPU
	cudaFree(d_M);
	cudaFree(d_MH);

	return 0;
}
