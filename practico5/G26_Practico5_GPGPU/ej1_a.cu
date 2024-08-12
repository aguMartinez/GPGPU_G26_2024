#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda.h"

#define MAX_THREADS_PER_BLOCK 1024


int* constantVector(int n, int c){
  int* V = (int*)malloc(n * sizeof(int));

  for (int i = 0; i < n; i++){
      V[i] = c;
  }
  return V;
}
void printVector(int* V, int n){
  for (int i = 0; i < n; i++){
      if((i % 512) == 0)
        printf("\n");
      printf("%d ", V[i]);
  }
  printf("\n");
}

__global__ void exclusiveScanKernel(int* d_out, int* d_in, int* d_sums, int n) {
  extern __shared__ int temp[];

  int thid = threadIdx.x;
  int offset = 1;
  int block_offset = blockIdx.x * n;

  temp[2 * thid] = d_in[block_offset + (2 * thid)];
  temp[2*thid+1] = d_in[block_offset + (2 * thid * + 1)];

  //parte 1: up-sweep
  for (int d = n/2; d > 0; d /= 2){
    __syncthreads();
    if(thid < d){
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  
  //ultimo elemento se pone en cero, antes de eso se carga la suma parcial
  if (thid == 0) {
    if(d_sums != NULL)
      d_sums[blockIdx.x] = temp[n - 1];
      temp[n - 1] = 0;
  }

  //down-sweep
  for (int d = 1; d < n; d *= 2){
    offset /= 2;
    __syncthreads();

    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();


  d_out[block_offset + (2 * thid)] = temp[2 * thid];
  d_out[block_offset + (2 * thid + 1)] = temp[2 * thid + 1];
}

__global__ void sumOffsetsKernel(int* d_out, int* d_offsets, int blockSize) {
  int thid = threadIdx.x;
  int block_offset = blockIdx.x * blockSize;

  int ai = 2 * thid;
  int bi = 2 * thid + 1;

  int offset = d_offsets[blockIdx.x];

  d_out[block_offset + ai] += offset;
  d_out[block_offset + bi] += offset;
}

int main(int argc, char *argv[]) {
  
  if (argc < 2){
    printf("Faltaron argumentos %d \n", argc);
    return 1;
  }

  // Único argumento de entrada: tamaño del vector
  int n = atoi(argv[1]);
  int blockSize = (n < MAX_THREADS_PER_BLOCK*2) ? n/2 : MAX_THREADS_PER_BLOCK;

  printf("n = %d\n", n);
  printf("tamaño bloque = %d\n", blockSize);
  
  // Vector de entrada:
  int* h_in = constantVector(n,1);
  int* h_out = (int*)malloc(n*sizeof(int));

  int* d_in;
  int* d_out;

  size_t size = n * sizeof(int);

  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);
  
  // Divido los bloques entre dos porque solo necesito la mitad de ellos (cada thread resuelve dos entradas)
  int blocks = (n + blockSize - 1) / blockSize / 2;
  size_t sharedSize =  2 * blockSize * sizeof(int);

  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);


  // [begin] evento
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  int* d_sums;
  cudaMalloc(&d_sums, blocks * sizeof(int));
  
  
  // El primer Scan devuelve las sumas parciales en d_sums
  exclusiveScanKernel<<<blocks, blockSize, sharedSize>>>(d_out, d_in, d_sums, 2*blockSize);


  // [begin] scan secuencial
  // de las sumas parciales:
  int* h_sums = (int*)malloc(blocks*sizeof(int));
  cudaMemcpy(h_sums, d_sums, blocks * sizeof(int), cudaMemcpyDeviceToHost);

  int* h_offsets = (int*)malloc(blocks*sizeof(int));
  h_offsets[0] = 0;
  
  for (int i = 1; i < blocks; i++) {
      h_offsets[i] = h_offsets[i - 1] + h_sums[i - 1];
  }

  // [end] scan secuencial

  // copiamos el resultado a memoria del device
  int* d_offsets;
  cudaMalloc(&d_offsets, blocks * sizeof(int));
  cudaMemcpy(d_offsets, h_offsets, blocks * sizeof(int), cudaMemcpyHostToDevice);

  // procedemos a sumar los offsets correspondientes a cada elemento, esta parte es totalmente paralelizable por lo que se hace en otro kernel:
  sumOffsetsKernel<<<blocks, blockSize>>>(d_out, d_offsets, 2*blockSize);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
   printf("Tiempo: %f ns\n", elapsedTime * 1e+6);
  // [end] evento
  
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
  
  //[debug] begin
  /*
  printVector(h_sums, blocks);
  printf("-------------------------\n");
  printVector(h_out, n);
  */
  //[debug] end
 

  free(h_in);
  free(h_out);
  free(h_sums);
  free(h_offsets);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_sums);
  cudaFree(d_offsets);
}