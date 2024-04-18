#include <stdio.h>
#include <stdlib.h>
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

void read_file(const char*, int*);
int get_text_length(const char * fname);

#define A 15
#define B 27
#define M 256
#define A_MMI_M -17

#define N 512


__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void decrypt_kernel(int *d_message, int length)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = index; i < length; i += blockDim.x * gridDim.x) {
		d_message[i] = modulo(A_MMI_M*(d_message[i] - B), M);
	}
}

void ej1A(int* d_message, int length) {
	int threads_per_block = 256;
	int blocks_per_grid = 1;

	decrypt_kernel<<<blocks_per_grid, threads_per_block>>>(d_message, length);
}

void ej1B(int* d_message, int length) {
	int threads_per_block = 256;
	int blocks_per_grid = (length + threads_per_block - 1) / threads_per_block;

	decrypt_kernel<<<blocks_per_grid, threads_per_block>>>(d_message, length);
}

void ej1C(int* d_message, int length) {
	int blocks_per_grid = 128;
	int threads_per_block = 256;

	decrypt_kernel<<<blocks_per_grid, threads_per_block>>>(d_message, length);
}

__global__ void countCharacters(int* d_message, int* histograma, int length) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < length; i += blockDim.x * gridDim.x) {
		atomicAdd( &(histograma[d_message[i]]), 1);
	}
}

int main(int argc, char *argv[])
{
	int *h_message;
	int *d_message;
	unsigned int size;

	const char * fname;

	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else
		fname = argv[1];

	int length = get_text_length(fname);

	size = length * sizeof(int);

	// reservar memoria para el mensaje
	h_message = (int *)malloc(size);

	// leo el archivo de la entrada
	read_file(fname, h_message);

	/* reservar memoria en la GPU */
	CUDA_CHK(cudaMalloc((void **)&d_message, size));

	/* copiar los datos de entrada a la GPU */

 	cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice);

	/* Configurar la grilla y lanzar el kernel */
	/* Descomentar la parte de ejercicio a ejecutar */
	//ej1A(d_message, length);
	//ej1B(d_message, length);
	ej1C(d_message, length);

	/* Copiar los datos de salida a la CPU en h_message */
 	cudaMemcpy(h_message, d_message, size, cudaMemcpyDeviceToHost);

	// despliego el mensaje

	for (int i = 0; i < length; i++) {
		printf("%c", (char)h_message[i]);
	}
	printf("\n");

    int* d_count;
    int h_count[256] = {0};
    size = 256 * sizeof(int);

    cudaMalloc((void**)&d_count, size);

    cudaMemcpy(d_count, h_count, size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (length + threads_per_block - 1) / threads_per_block;

	countCharacters<<<blocks_per_grid, threads_per_block>>>(d_message, d_count, length);

    cudaMemcpy(h_count, d_count, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 256; i++) {
        if (i >= 32 && i <= 126) {
            printf("%c %d \n", i, h_count[i]);
        } else {
            printf("0x%02x %d \n", i, h_count[i]);
		}    
	}


	// libero la memoria en la CPU
	free(h_message);

	// libero la memoria en la GPU
	cudaFree(d_message);
    cudaFree(d_count);

	return 0;
}


int get_text_length(const char * fname)
{
	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags

	size_t pos = ftell(f);    
	fseek(f, 0, SEEK_END);    
	size_t length = ftell(f); 
	fseek(f, pos, SEEK_SET);  

	fclose(f);

	return length;
}

void read_file(const char * fname, int* input)
{
	// printf("leyendo archivo %s\n", fname );

	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find %s file \n", fname);
		exit(1);
	}

	//fread(input, 1, N, f);
	int c; 
	while ((c = getc(f)) != EOF) {
		*(input++) = c;
	}

	fclose(f);
}
