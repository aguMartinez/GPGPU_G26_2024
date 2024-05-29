/* Ejercicio 3 | Practico 4 | Grupo 26 | GPGPU 2024 */

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define HISTOGRAM_LENGTH 4
#define FILAS 8
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
	extern __shared__ int shared_histogram[];
	
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

__global__ void reduction(int* d_MH, int numRows, int numCols) {
	extern __shared__ int intermedio[];

	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int i = blockDim.x * blockDim.y / 2;
	int num_pos = threadIdx.y * blockDim.x + threadIdx.x;
	//int block_pos = blockIdx.y * gridDim.x + blockIdx.x;

	if (num_pos < i) {
		intermedio[num_pos] = d_MH[y * HISTOGRAM_LENGTH + x];
    }
    __syncthreads();

	

	while ( i != 0) {
		if (num_pos < i) {
			printf("(%d,%d) (%d,%d) %d + %d \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, intermedio[num_pos], d_MH[y * HISTOGRAM_LENGTH + x + i]);
			
			intermedio[num_pos] = intermedio[num_pos] + d_MH[y * HISTOGRAM_LENGTH + x + i];
			if(x == 0 && y == 0);
			
		}
		__syncthreads();
		i = i / 2;
	}

/*
	if (blockIdx.y == 0) {
		sdata[num_pos] = d_MH[block_pos * HISTOGRAM_LENGTH + num_pos] + d_MH[(block_pos + 1) * HISTOGRAM_LENGTH + num_pos];
	}
*/
    __syncthreads();

    // Guardar los resultados en una posición adecuada
    if (num_pos < HISTOGRAM_LENGTH) {
        d_MH[y * HISTOGRAM_LENGTH + x] = intermedio[num_pos];
    }
}


/*
__global__ void reduction(float * output, float * input){
	extern __shared__ float intermedio[];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	intermedio[threadIdx.x] = input[idx];
	__syncthreads();
	
	int i = blockDim.x/2;

	while (i != 0) {
		if (threadIdx.x < i)
			intermedio[threadIdx.x] = intermedio[threadIdx.x] + intermedio[threadIdx.x + i];
	
		__syncthreads();
		i = i / 2;
	}
	__syncthreads();
	if (threadIdx.x==0){
		output[blockIdx.x] = intermedio[0];
	}
}
*/


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

	int sizeShared = blockX * blockY / 2;
	int sizeMH = numBlocks.x * numBlocks.y * HISTOGRAM_LENGTH * sizeof(int);

	int* h_MH = (int*) malloc(sizeMH);
	int* d_MH;

	CUDA_CHK(cudaMalloc((void **)&d_MH, sizeMH));

    decrypt_kernel_ej3B<<<numBlocks, threadsPerBlock, sizeShared>>>(d_M, d_H, d_MH);

	/* Copiar los datos de salida */
 	// cudaMemcpy(h_H, d_H, sizeH, cudaMemcpyDeviceToHost);
 	cudaMemcpy(h_MH, d_MH, sizeMH, cudaMemcpyDeviceToHost);

    printf("-----------MATRIZ ORIGINAL------------\n");
	printMatrix(h_M, FILAS, COLUMNAS);

	int currentRows = numBlocks.x * numBlocks.y;  // Número inicial de histogramas
	//int nextRows = currentRows;

	cudaMemcpy(h_MH, d_MH, sizeMH, cudaMemcpyDeviceToHost);

    printf("-----------MATRIZ HISTOGRAMA ANTES DEL REDUCE------------\n");
	printMatrix(h_MH, currentRows, HISTOGRAM_LENGTH);


	blockX = HISTOGRAM_LENGTH;
	blockY = 4;

	dim3 threadsPerBlock2(blockX, 4);
	dim3 numBlocks2( (HISTOGRAM_LENGTH + blockX - 1) / blockX, (currentRows + blockY - 1) / blockY);

	reduction<<<numBlocks2, threadsPerBlock2>>>(d_MH, currentRows, HISTOGRAM_LENGTH);
	/*cudaDeviceSynchronize();

	dim3 threadsPerBlock2(HISTOGRAM_LENGTH, 2);
	dim3 numBlocks2( (HISTOGRAM_LENGTH * 2 - 1) / HISTOGRAM_LENGTH, (currentRows + 2 - 1) / 2);

	reduction<<<numBlocks2, threadsPerBlock2>>>(d_MH, currentRows, HISTOGRAM_LENGTH);
*/
 	cudaMemcpy(h_MH, d_MH, sizeMH, cudaMemcpyDeviceToHost);

    printf("-----------MATRIZ HISTOGRAMA LUEGO DEL REDUCE------------\n");
	printMatrix(h_MH, currentRows, HISTOGRAM_LENGTH);

    // printMatrix(h_M, FILAS, COLUMNAS);

/*
    printMatrix(h_M, FILAS, COLUMNAS);
    printf("-----------------------\n");
	printMatrix(h_H, 1, HISTOGRAM_LENGTH);
*/
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
