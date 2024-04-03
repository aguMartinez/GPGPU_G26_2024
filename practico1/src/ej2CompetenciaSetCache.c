//
// Created by amartinez on 1/4/2024.
//

#include "../include/ej2CompetenciaSetCache.h"
#include "../include/ej2Blocking.h"
#include "../include/utils.h"

#include <stdio.h>
#include <time.h>

void multiplicarMatricesBlockingRectangular(int** A, int** B, int** C, int n, int bsizeX, int bsizeY)
{
    int i, j, k, kk, jj;
    volatile int sum;
    int enX = bsizeX * (n/bsizeX); // Ajustar para bloques en dirección X
    int enY = bsizeY * (n/bsizeY); // Ajustar para bloques en dirección Y

    // Inicializar C a 0
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            C[i][j] = 0;

    // Bucle principal para multiplicar bloques
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

    int n = 64 * 16 * 2; //2048
    int bloque = 12 * 4;


    int** A = sequentialMatrix(n);
    int** B = sequentialMatrix(n);
    int** C = initializeMatrix(n);


    double interval = 0;
    NS(multiplicarMatricesBlocking(A,B,C,n,bloque), interval)

    printf("tiempo (s)\n");
    printf("%f",interval/1E9);

}