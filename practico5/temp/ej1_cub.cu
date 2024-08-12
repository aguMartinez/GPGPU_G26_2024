#include <iostream>

#include <cuda.h>
#include <cub/cub.cuh>
using namespace std;

// CustomMin functor
struct Add
{
  template <typename T>
  //CUB_RUNTIME_FUNCTION forceinline

  __host__ __device__ T operator()(const T &a, const T &b) const {
    return a+b;
  }
};

int* constantVector(int n, int c){
  int* V = (int*)malloc(n * sizeof(int));

  for (int i = 0; i < n; i++){
      V[i] = c;
  }
  return V;
}

int main (int argc, char * argv[]) {
  clock_t start_t = clock ();
  
  if (argc < 2) {
  printf("Number of elements missing! \n");
  return 1;
  }

  /* definir tamanio del vector*/
  int n = atoi(argv[1]);

  int* a = constantVector(n, 1);
  int* b = new int [n];
  int* dev_a, * dev_b;
  Add add_op;

  cudaMalloc ((void**) & dev_a, n*sizeof(int));
  cudaMalloc ((void**) & dev_b, n*sizeof(int));

  printf("Arreglo original: \n");
  for (int i = 0; i < n; i ++) 		
    printf("%d ", a[i]);
  
  printf("\n");

  cudaMemcpy (dev_a, a, n*sizeof(int), cudaMemcpyHostToDevice);	

  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, dev_a, dev_b, add_op, n);

  // Allocate temporary storage for inclusive prefix scan
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, dev_a, dev_b, add_op, n);
  cudaMemcpy (b, dev_b, n*sizeof(int), cudaMemcpyDeviceToHost);	


  printf("Arreglo pos-scan: \n");
  for (int i = 0; i < n; i ++) 
    printf("%d ", b[i]);
  printf("\n");

  cudaFree (dev_a);
  cudaFree (dev_b);

  clock_t end_t = clock ();
  cout << "time = " << (double)(end_t - start_t) / CLOCKS_PER_SEC << endl;

  return 1;
}