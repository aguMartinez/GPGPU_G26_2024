#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda.h"

#define NS(f,elap)                                                                                           \
        {                                                                                                    \
        struct timespec t_ini,t_fin;                                                                         \
            clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                          \
            f;                                                                                               \
            clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                          \
            elap = 1000000000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec);             \
        }



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



int main(int argc, char *argv[]) {
  
  if (argc < 2){
    printf("Faltaron argumentos %d \n", argc);
    return 1;
  }

  // Único argumento de entrada: tamaño del vector
  int n = atoi(argv[1]);

  printf("n = %d\n", n);

  // Vector de entrada:
  int* h_in = constantVector(n,1);
  int* h_out = (int*)malloc(n*sizeof(int));

  double interval;

  NS(
  
    h_out[0] = 0;
    for (int i = 1; i < n; i++)
        h_out[i] = h_out[i - 1] + h_in[i - 1];
  
  , interval);

  //[debug] begin
  printf("interval = %f ns", interval);
  //printVector(h_out, n);
  //[debug] end
 

  free(h_in);
  free(h_out);
}