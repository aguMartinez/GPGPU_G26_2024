#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda.h"
#include <cub/cub.cuh>

#define MS(f,elap)                                                                                           \
{                                                                                                            \
struct timespec t_ini,t_fin;                                                                                 \
    clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                                  \
    f;                                                                                                       \
    clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                                  \
    elap = 1000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec)/1000000.0;                 \
}

int* constantVector(int n){
  int* V = (int*)malloc(n * sizeof(int));

  for (int i = 0; i < n; i++){
      V[i] = 1;
  }
  return V;
}

void printVector(int* V, int n){
  for (int i = 0; i < n; i++){
      printf("%d ", V[i]);
  }
  printf("\n");
}




/* 
 * Implentación basada en documentación oficial:
 * https://github.com/dmlc/cub/blob/master/cub/device/device_scan.cuh
*/

int main(int argc, char *argv[]) {
  
  if (argc < 2){
    printf("Faltaron argumentos %d \n", argc);
    return 1;
  }
  
 
  int num_items = atoi(argv[1]);
  printf("num_items = %d\n", num_items);
  
  int* host_data = constantVector(num_items);
  
  //variables de dispositivo
  

  
  int* d_in;
  int* d_out;
  
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;




  cudaMalloc(&d_in, num_items * sizeof(int));
  cudaMalloc(&d_out, num_items * sizeof(int));
  cudaMemcpy(d_in, host_data, num_items * sizeof(int), cudaMemcpyHostToDevice);

  double interval = 0;
  MS(
  //primera llamada para calcular el tamaño del espacio temporal
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);


  , interval);
  cudaMemcpy(host_data, d_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);



  //liberar memoria
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_temp_storage);

  /*
  printVector(host_data, num_items);
  */
  printf("Interval: %f\n", interval);




  free(host_data);

  return 0;

}