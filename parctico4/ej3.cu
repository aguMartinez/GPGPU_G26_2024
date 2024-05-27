/* Ejercicio 3 | Practico 4 | Grupo 26 | GPGPU 2024 */

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define HISTOGRAM_LENGTH 256
#define FILAS 3840
#define COLUMNAS 2160

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

__global__ void decrypt_kernel_ej3(int *d_M, int * d_H)
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
	atomicAdd(&shared_histogram[elem], 1);
	
	__syncthreads();

	if (num_pos < HISTOGRAM_LENGTH) {
		atomicAdd( &d_H[num_pos], shared_histogram[num_pos]);
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

	int * h_M = randomMatrix(FILAS, COLUMNAS);
	int *h_H = (int*) malloc(sizeH);

	memset(h_H, 0, sizeH);

	printMatrix(h_H, 1, HISTOGRAM_LENGTH);

	int * d_M; 
	int * d_H;

	/* reservar memoria en la GPU */
	CUDA_CHK(cudaMalloc((void **)&d_H, sizeH));
	CUDA_CHK(cudaMalloc((void **)&d_M, sizeM));

	/* copiar los datos de entrada a la GPU */

 	cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);
 	cudaMemcpy(d_H, h_H, sizeH, cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(blockX, blockY);
	dim3 numBlocks( (COLUMNAS + blockX - 1) / blockX, (FILAS + blockY - 1) / blockY);

	decrypt_kernel_ej3<<<numBlocks, threadsPerBlock>>>(d_M, d_H);

	/* Copiar los datos de salida a la CPU en h_message */
 	cudaMemcpy(h_H, d_H, sizeH, cudaMemcpyDeviceToHost);


    printMatrix(h_M, FILAS, COLUMNAS);
    printf("-----------------------\n");

	printMatrix(h_H, 1, HISTOGRAM_LENGTH);

	// libero la memoria en la CPU
	free(h_M);
	free(h_H);

	// libero la memoria en la GPU
	cudaFree(d_M);
    cudaFree(d_H);

	return 0;
}
