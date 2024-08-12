#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda.h"

#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main(int argc, char *argv[]) {
  
  if (argc < 2){
    printf("Faltaron argumentos %d \n", argc);
    return 1;
  }

  int n = atoi(argv[1]);
  printf("num_items = %d\n", n);


  
  thrust::host_vector<int> h_vec(n);

  for(int i = 0; i < n; i++)
    h_vec[i] = 1;


  thrust::device_vector<int> d_vec = h_vec;

  // [begin] evento
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  

  double interval = 0;

  thrust::exclusive_scan(d_vec.begin(), d_vec.end(), d_vec.begin());
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Tiempo: %f ns\n", elapsedTime * 1e+6);
  // [end] evento

  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
  

  printf("THRUST:\n");
  //imprimir
  /*
  for (int i = 0; i < n; i++){
      printf("%d ", h_vec[i]);
  }
  */
  
  return 0;
}