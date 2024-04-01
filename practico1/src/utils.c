//
// Created by amartinez on 20/3/2024.
//

#include <stdlib.h>
#include <stdio.h>
#include "../include/utils.h"
#include <sched.h>
#include <Windows.h>

int ** randomMatrix(int n){
    int **a = (int **) _aligned_malloc(n * sizeof(int *), 64);

    for (int i = 0; i < n; i++) {
        a[i] = (int *) _aligned_malloc(n * sizeof(int), 64);
        for (int j = 0; j < n; j++) {
            a[i][j] = rand();
        }
    }
    return a;
}

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
