#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda.h"

#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define MS(f,elap)                                                                                           \
{                                                                                                            \
struct timespec t_ini,t_fin;                                                                                 \
    clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                                  \
    f;                                                                                                       \
    clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                                  \
    elap = 1000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec)/1000000.0;                 \
}

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
  
  double interval = 0;
  MS(
  thrust::exclusive_scan(d_vec.begin(), d_vec.end(), d_vec.begin());
  , interval);

  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
  


  //imprimir
  /*
  for (int i = 0; i < n; i++){
      printf("%d ", h_vec[i]);
  }
  */
  printf("Interval: %f\n", interval);

  return 0;
}