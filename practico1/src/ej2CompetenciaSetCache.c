
#include "../include/ej2CompetenciaSetCache.h"
#include "../include/ej2Blocking.h"
#include "../include/utils.h"

#include <stdio.h>
#include <time.h>

void multiplicarMatricesBlockingRectangular(int** A, int** B, int** C, int n, int bsizeX, int bsizeY)
{
    int i, j, k, kk, jj;
    volatile int sum;
    int enX = bsizeX * (n/bsizeX);
    int enY = bsizeY * (n/bsizeY);

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            C[i][j] = 0;

    for (kk = 0; kk < enX; kk += bsizeX) {
        for (jj = 0; jj < enY; jj += bsizeY) {
            for (i = 0; i < n; i++) {
                for (j = jj; j < jj + bsizeY; j++) {
                    sum = C[i][j];
                    for (k = kk; k < kk + bsizeX; k++) {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
        }
    }
}

void ej2CompetenciaSetCache() {

    int n = 64 * 16 * 2;
    int bsizeColumna = n / 4;
    int bsizeFila = 64*16;
    int bloque = 128;

    int** A = sequentialAlignedMatrix(n);
    int** B = sequentialAlignedMatrix(n);
    int** C = initializeMatrix(n);

    double interval = 0;
    double gflops = 0;

    NS(multiplicarMatricesBlockingRectangular(A,B,C,n,bsizeFila,bsizeColumna), interval)

    gflops = ((2.0 * n * n * n) / (interval / 1E9))/ 1E9;
    printf("tiempo (s),Gflops\n");
    printf("%f %f\n",interval/1E9, gflops);

    freeAlignedMatrix(A,n);
    freeAlignedMatrix(B,n);
    freeMatrix(C,n);

    A = sequentialAlignedMatrix(n);
    B = sequentialAlignedMatrix(n);
    C = initializeMatrix(n);

    NS(multiplicarMatricesBlocking(A,B,C,n,bloque), interval)

    gflops = ((2.0 * n * n * n) / (interval / 1E9))/ 1E9;
    printf("tiempo (s),Gflops\n");
    printf("%f %f\n",interval/1E9, gflops);

    freeAlignedMatrix(A,n);
    freeAlignedMatrix(B,n);
    freeMatrix(C,n);

}