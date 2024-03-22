//
// Created by amartinez on 20/3/2024.
//

#include <stdlib.h>
#include "utils.h"

int ** randomMatrix(int n){
    int **a = malloc(n * sizeof(int *));

    for (int i = 0; i < n; i++) {
        a[i] = malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            a[i][j] = rand();
        }
    }
    return a;
}

int* randomArray(int n){
    int* a = malloc(n * sizeof(int *));

    for (int i = 0; i < n; i++) {
        a[i] = rand();
    }
    return a;
}


void freeMatrix(int ** a, int n) {
    for (int i = 0; i < n; i++) {
        free(a[i]);
    }
    free(a);
}


//void setCPU{
//    DWORD_PTR processAffinityMask = 1;
//
//    HANDLE hProcess = GetCurrentProcess();
//
//    if (SetProcessAffinityMask(hProcess, processAffinityMask) == 0) {
//        printf("Error al establecer la máscara de afinidad del proceso. Codigo de error: %lu\n", GetLastError());
//    } else {
//        printf("La mascara de afinidad del proceso se establecio correctamente.\n");
//    }
//}