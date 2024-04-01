//
// Created by Valen on 26/3/2024.
//

#include "../include/utils.h"
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

    for (int i = 0; i < m * n; i++) {
        C[i] = 0;
    }

    for (int row = 0; row < m; row++) {
        for (int it = 0; it < p; it++) {
            for (int col = 0; col < n; col++) {
                C[row * n + col] += A[row * p + it] * B[it * n + col];
            }
        }
    }
}





int ej2Reordenamiento() {

    if (freopen("output_ej2_reordenado.csv", "w", stdout) == NULL) {
        perror("freopen failed");
    }

    printf("m, p, n, Tiempo reordenado (ns), Tiempo no reordenado (ns), GFlops reordenado, GFlops no reordenado\n");

    int tope = 5000;
    int m, p, n;

    int* A;
    int* B;
    int* C;

    double elapNoReordenado, gflopsNoReordenado, elapReordenado, gflopsReordenado;

    for (int i = 2600; i <= tope; i+=100) {

        m = p = n = i;
        A = randomArray(m * p);
        B = randomArray(p * n);
        C = (int *) malloc(m * n * sizeof(int));

        elapNoReordenado = 0;
        NS(multiplicarMatricesNoReordenado(A, B, C, m, p, n), elapNoReordenado);
        gflopsNoReordenado = (2.0 * m * p * n) / (elapNoReordenado * 1e-9);


        elapReordenado = 0;
        NS(multiplicarMatricesReordenado(A, B, C, m, p, n), elapReordenado);
        gflopsReordenado = (2.0 * m * p * n) / (elapReordenado * 1e-9);

        free(A);
        free(B);
        free(C);

        printf("%d, %d, %d, %f, %f, %f, %f\n", m, p, n, elapReordenado, elapNoReordenado, gflopsReordenado, gflopsNoReordenado);

    }

    return 0;
}