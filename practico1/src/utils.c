//
// Created by amartinez on 20/3/2024.
//

#include <stdlib.h>
#include <stdio.h>
#include "../include/utils.h"
#include <sched.h>
#include <Windows.h>

/* UTILIDADES DE MATRICES */

int** initializeMatrix(int n){
    int** A = (int **) malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++)
        A[i] = (int *) malloc(n * sizeof(int));
    return A;
}

int** sequentialMatrix(int n){
    int **A = (int **) malloc(n * sizeof(int*));

    for (int i = 0; i < n; i++) {
        A[i] = (int *) malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            A[i][j] = i*n + j;
        }
    }
    return A;
}

int** sequentialAlignedMatrix(int n){
    int **A = (int **) _aligned_malloc(n * sizeof(int*),32*1024);

    for (int i = 0; i < n; i++) {
        A[i] = (int *) _aligned_malloc(n * sizeof(int),32*1024);
        for (int j = 0; j < n; j++) {
            A[i][j] = i*n + j;
        }
    }
    return A;
}

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

void printMatrix(int** A, int n){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", A[i][j]);
        }
        printf("\n");
    }
}

int** cloneMatrix(int** A, int n){
    int** B = initializeMatrix(n);
    for(int i = 0; i < n;i++)
        for (int j = 0; j < n; j++)
            B[i][j] = A[i][j];
    return B;
}

/* UTILIDADES DE ARREGLOS */

int* randomArray(int n){

    int* a = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        a[i] = rand();
    }
    return a;
}

int* sequentialArray(int n){

    int* a = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        a[i] = i;
    }
    return a;
}

void freeMatrix(int ** a, int n) {
    for (int i = 0; i < n; i++)
        free(a[i]);

    free(a);
}

void freeAlignedMatrix(int ** a, int n) {
    for (int i = 0; i < n; i++)
        _aligned_free(a[i]);

    _aligned_free(a);
}


void shuffleArray(int* array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}


void setConsoleAsStdOutput(){
#ifdef _WIN32
    if (freopen("CON", "w", stdout) == NULL) {
        perror("freopen failed");
    }
#else
    if (freopen("/dev/tty", "a", stdout) == NULL) {
        perror("freopen failed");
        return -1;
    }
#endif
}

void set_cpu_affinity(int cpu_id) {
#ifdef _WIN32
    DWORD_PTR mask = 1 << cpu_id;
    HANDLE process = GetCurrentProcess();

    if (!SetProcessAffinityMask(process, mask)) {
        fprintf(stderr, "Error al establecer la afinidad de CPU: %lu\n", GetLastError());
    }
#else
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);

    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        perror("sched_setaffinity failed");
    }
#endif
}
