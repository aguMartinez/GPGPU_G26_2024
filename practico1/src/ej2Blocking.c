//
// Created by amartinez on 1/4/2024.
//

#include "../include/utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


/* Utiliza array bidimensional para simplificar la escritura de indices
 * C es un output
 * */
void multiplicarMatricesBlocking(int** A, int** B, int** C, int n, int bsize)
{
    int i, j, k, kk, jj;
    volatile int sum;
    int en = bsize * (n/bsize); /* Amount that fits evenly into blocks */

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            C[i][j] = 0;

    for (kk = 0; kk < en; kk += bsize) {
        for (jj = 0; jj < en; jj += bsize) {
            for (i = 0; i < n; i++) {
                for (j = jj; j < jj + bsize; j++) {
                    sum = C[i][j];
                    for (k = kk; k < kk + bsize; k++) {
                        sum += A[i][k]*B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
        }
    }
}

int ej2Blocking(int MAX_SIZE){

    if (freopen("output_ej2_blocking.csv", "w", stdout) == NULL) {
        perror("freopen failed");
    }

    printf("n,b,tiempo (ms)\n");

    int tope = 2048;

    for (int n=2;n<=tope;n=n*2){

        for(int b = 2; b<=n; b=b*2){
            int** A = randomMatrix(n);
            int** B = randomMatrix(n);
            int** C = randomMatrix(n); //Se inicializa con basura (va a ser sustituido)

            double interval = 0;
            NS(multiplicarMatricesBlocking(A, B, C, n, b), interval)

            printf("%d,%d,%f\n",n,b,interval/1e+6);
            freeMatrix(A,n);
            freeMatrix(B,n);
            freeMatrix(C,n);
        }

    }
    setConsoleAsStdOutput();
    printf("Finalizado! Resultados guardados en output_ej2_blocking.csv\n");
    return 1;
}