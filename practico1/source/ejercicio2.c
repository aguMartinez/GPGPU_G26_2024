//
// Created by Valen on 26/3/2024.
//

#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void multiplicarMatricesNoReordenado(int* A, int* B, int* C, int m, int p, int n){
    int sum;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            sum = 0;
            for (int it = 0; it < p; it++) {
                sum += A[row * p + it] * B[it * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

void multiplicarMatricesReordenado(int* A, int* B, int* C, int m, int p, int n){
    int sum;
    for (int col = 0; col < n; col++)
        for (int row = 0; row < m; row++){

            sum=0;
            for (int it = 0; it < p; it++)
                sum += A[row * p + it] * B[it * n + col];
            C[row * n + col]=sum;
        }
}

int ejercicio2() {
    int m = 1000;
    int p = 1000;
    int n = 1000;

    int* A = randomArray(m * p);
    int* B = randomArray(p * n);
    int* C = malloc(m * n * sizeof(int));

    double elapNoReordenado = 0;
    NS(multiplicarMatricesNoReordenado(A, B, C, m, p, n), elapNoReordenado);
    double gflopsNoReordenado = (2.0 * m * p * n) / (elapNoReordenado) * 1e-9;
    printf("Tiempo multiplicarMatricesNoReordenado: %f ns, GFLOPS: %f\n", elapNoReordenado, gflopsNoReordenado);

    double elapReordenado = 0;
    NS(multiplicarMatricesReordenado(A, B, C, m, p, n), elapReordenado);
    double gflopsReordenado = (2.0 * m * p * n) / (elapReordenado) * 1e-9;
    printf("Tiempo multiplicarMatricesReordenado: %f ns, GFLOPS: %f\n", elapReordenado, gflopsReordenado);

    free(A);
    free(B);
    free(C);

    return 0;
}