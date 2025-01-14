#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda.h"

int* constantVector(int n, int c){
  int* V = (int*)malloc(n * sizeof(int));

  for (int i = 0; i < n; i++){
      V[i] = c;
  }
  return V;
}

int* sequentialVector(int n){
  int* V = (int*)malloc(n * sizeof(int));

  for (int i = 0; i < n; i++){
      V[i] = i;
  }
  return V;
}

void printVector(int* V, int n){
  for (int i = 0; i < n; i++){
      printf("%d ", V[i]);
  }
  printf("\n");
}

__global__ void exclusiveScanKernel(int* d_out, int* d_in, int n) {
  extern __shared__ int temp[];

  int thid = threadIdx.x;
  int offset = 1;

  temp[2 * thid] = d_in[2 * thid];
  temp[2*thid+1] = d_in[2*thid+1];
  //parte 1: up-sweep
  for (int d = n/2; d > 0; d /= 2){
    if(thid < d){
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  
  //ultimo elemento se pone en cero
  if (thid == 0) {
    temp[n - 1] = 0;
  }

  //down-sweep
  for (int d = 1; d < n; d *= 2){
    offset /= 2;
    __syncthreads();

    //[debug] begin
    //printf("[%d, %d] %d \n", d, thid, temp[thid]);
    //[debug] end

    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();


  d_out[2 * thid] = temp[2 * thid];
  d_out[2 * thid + 1] = temp[2 * thid + 1];
  
  //[debug] begin
  //printf("[%d] %d \n", thid, temp[thid]);
  //[debug] end

}


int main(int argc, char *argv[]) {
  if (argc < 3){
    printf("Faltaron argumentos %d \n", argc);
    return 1;
  }

  /* definir tamanios de matriz*/
  int n = atoi(argv[1]);
  int blockX = atoi(argv[2]);

  for (int input = 1; input < 3; input++)
  {
    printf("%d ", atoi(argv[input]));
  }
  printf("\n");
  /* genero vector de entrada*/

  int* h_in = constantVector(n,1);
  int* h_out = (int*)malloc(n*sizeof(int));


  int* d_in;
  int* d_out;
  size_t size = n * sizeof(int);

  // Asignar memoria en el dispositivo
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // Copiar datos de entrada al dispositivo
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  // Configurar el tamaño de los bloques e hilos
  int threadsPerBlock = blockX;
  int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  size_t shared_mem_size =  n * sizeof(int);

  // Lanzar el kernel
  //exclusive_scan_kernel<<<blocks, threads, shared_mem_size>>>(d_out, d_in, n);
  exclusiveScanKernel<<<blocks, threadsPerBlock, shared_mem_size>>>(d_out, d_in, n);

  // Copiar los datos de salida al host
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  // Imprimir datos
  printVector(h_in, n);
  printVector(h_out, n);

  // Liberar la memoria del dispositivo
  cudaFree(d_in);
  cudaFree(d_out);
}