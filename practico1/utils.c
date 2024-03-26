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
    int* a = malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        a[i] = rand();
    }
    return a;
}

int* sequentialArray(int n){
    int* a = malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        a[i] = i;
    }
    return a;
}

void freeMatrix(int ** a, int n) {
    for (int i = 0; i < n; i++) {
        free(a[i]);
    }
    free(a);
}

void shuffleArray(int* array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

struct cacheInicial cacheInicial(int *pInt, void *pVoid, void *pVoid1) {
    struct cacheInicial result;
    result.L1 = pInt;
    result.L2 = pVoid;
    result.L3 = pVoid1;
    return result;
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