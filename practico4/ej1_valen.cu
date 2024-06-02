/* Ejercicio 1 | Practico 4 | Grupo 26 | GPGPU 2024*/

#include <stdlib.h>
#include <stdio.h>
#include "cuda.h"

#define TILE_FILA 16
#define TILE_COLUMNA 32

#define CUDA_CHK(ans)                         \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

int *randomMatrix(int f, int c)
{
    int *A = (int *)malloc(f * c * sizeof(int));

    for (int i = 0; i < f * c; i++)
    {
        A[i] = rand() % 100;
    }
    return A;
}

int isTransponse(int * m1, int * m2, int c, int f) {

    for (int i = 0; i < f; i++) {
        for (int j = 0; j < c; j ++) {
            if (m1[i * c + j] != m2[j*f + i]) {
                return 0;
            }
        }
    }
    return 1;
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

__global__ void transposeKernel(int *d_M, int *d_MTrans, int f, int c)
{
    __shared__ int tile[TILE_COLUMNA * TILE_FILA];

    int x = blockIdx.x * TILE_COLUMNA + threadIdx.x;
    int y = blockIdx.y * TILE_FILA + threadIdx.y;
    // Cargar en tile de manera transpuesta

    if (x < c && y < f)
        tile[threadIdx.x * TILE_FILA + threadIdx.y] = d_M[y * c + x];

    __syncthreads();

    int num_pos = threadIdx.y * TILE_COLUMNA + threadIdx.x;

    int offset = blockIdx.x * TILE_COLUMNA * f + blockIdx.y * TILE_FILA ;

    int cuantoMePasoColumna = num_pos / TILE_FILA;
    int cuantoMePasoFila = num_pos % TILE_FILA;

    int posTile = cuantoMePasoColumna * f + cuantoMePasoFila;

    if (num_pos < TILE_COLUMNA * TILE_FILA && offset + posTile < f*c)
        d_MTrans[offset + posTile] = tile[num_pos];
}

    /*
    int a_transponer = threadIdx.y * TILE_COLUMNA + threadIdx.x;
    int k = a_transponer / TILE_COLUMNA;
    int l = a_transponer % TILE_COLUMNA;
    */

    // Se invierten los indices de los bloques, pues es es el bloque transpuesto.
    /*
    int x_trans = blockIdx.y * TILE_COLUMNA + threadIdx.x;
    int y_trans = blockIdx.x * TILE_FILA + threadIdx.y;

    int n_esimo_elemento = threadIdx.y * TILE_COLUMNA + threadIdx.x;
    int k = n_esimo_elemento / TILE_COLUMNA;
    int l = n_esimo_elemento % TILE_COLUMNA;

    if (j < f && i < c)  // Ajustar los limites para la matriz transpuesta
    {
        // Escribir por fila en la memoria global

        d_MTrans[y_trans * c + x_trans] = tile[l * TILE_FILA + k];
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
    int f = atoi(argv[1]);
    int c = atoi(argv[2]);

    for (int input = 1; input < 3; input++)
    {
        printf("%d ", atoi(argv[input]));
    }

    printf("\n");

    int size = f * c * sizeof(int);

    /* generar matrices del host */
    int *h_M = randomMatrix(f, c);
    int *h_MTrans = (int *)malloc(size);

    /* reservar memoria en la GPU */
    int *d_M;
    int *d_MTrans;
    CUDA_CHK(cudaMalloc((void **)&d_M, size));      // matriz de entrada
    CUDA_CHK(cudaMalloc((void **)&d_MTrans, size)); // matriz de salida

    /* copiar los datos de entrada a la GPU */
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);

    /* Ej 1:*/
    dim3 threadsPerBlock(TILE_COLUMNA, TILE_FILA);
    dim3 numBlocks((c + TILE_COLUMNA - 1) / TILE_COLUMNA, (f + TILE_FILA - 1) / TILE_FILA);
    printf("numblocks: %d %d\n", numBlocks.x, numBlocks.y);

    for (int i = 0; i < 10; i++)
    {
        transposeKernel<<<numBlocks, threadsPerBlock>>>(d_M, d_MTrans, f, c);
    }

    /* Copiar los datos de salida a la CPU en h_message */
    cudaMemcpy(h_MTrans, d_MTrans, size, cudaMemcpyDeviceToHost);

    /* Imprimir resultados*/

    printf("trapureeesuhkfdkk %d \n", isTransponse(h_M, h_MTrans, c, f));

    printMatrix(h_M, f, c);
    printf("-----------------------\n");
    printMatrix(h_MTrans, c, f);

    /* Liberar memoria */
    free(h_M);
    free(h_MTrans);
    cudaFree(d_M);
    cudaFree(d_MTrans);

    return 1;
}
