#include <stdlib.h>
#include <stdio.h>

int ** randomMatrix(int n){
    int **A = (int **) malloc(n * sizeof(int*));

    for (int i = 0; i < n; i++) {
        A[i] = (int *) malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            A[i][j] = rand();
        }
    }
    return A;
}

__global__ void transpose_kernel(int *d_M, int* d_MT)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y

    if (x < n && y < n){
        AT[y + x * n] = A[x + y * n];
    }
}


 int main(){
    n = 256;
    m = n;

    int size = n*m*sizeof(int)
    int* M;
    int* R
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_MT, size);

    <<<transpose_kernel>>>(d_M, d_MT, size)
    return 1;
 }