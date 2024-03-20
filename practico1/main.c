#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <windows.h>

#define MS(f,elap)                                                                                           \
        double elap=0;                                                                                       \
        {                                                                                                    \
        struct timespec t_ini,t_fin;                                                                         \
            clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                          \
            f;                                                                                               \
            clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                          \
            elap = 1000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec)/1000000.0;         \
        }

int ** createMatrix(int n){
    int **a = malloc(n * sizeof(int *));

    for (int i = 0; i < n; i++) {
        a[i] = malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            a[i][j] = rand();
        }
    }

    return a;
}

void setCPU(){
    DWORD_PTR processAffinityMask = 1;

    HANDLE hProcess = GetCurrentProcess();

    if (SetProcessAffinityMask(hProcess, processAffinityMask) == 0) {
        printf("Error al establecer la máscara de afinidad del proceso. Código de error: %lu\n", GetLastError());
    } else {
        printf("La mascara de afinidad del proceso se establecio correctamente.\n");
    }
}

void freeMatrix(int ** a, int n) {
    for (int i = 0; i < n; i++) {
        free(a[i]);
    }
    free(a);
}

int main() {

    setCPU();
    int n = 5000;
    int ** a;

    for (int i = 1; i <= n; i++) {

        MS(a = createMatrix(i), interval);
        freeMatrix(a, i);
        printf("%d,%f  \n", i,interval);

    }
    return 0;
}

