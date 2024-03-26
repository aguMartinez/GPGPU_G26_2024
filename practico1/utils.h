//
// Created by amartinez on 20/3/2024.
//

#ifndef PRACTICO1_UTILS_H
#define PRACTICO1_UTILS_H

#define MS(f,elap)                                                                                           \
        double elap=0;                                                                                       \
        {                                                                                                    \
        struct timespec t_ini,t_fin;                                                                         \
            clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                          \
            f;                                                                                               \
            clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                          \
            elap = 1000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec)/1000000.0;         \
        }

#define NS(f,elap)                                                                                           \
        double elap=0;                                                                                       \
        {                                                                                                    \
        struct timespec t_ini,t_fin;                                                                         \
            clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                          \
            f;                                                                                               \
            clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                          \
            elap = 1000000000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec);         \
        }

#define kb 1000 //1 kb = 1000b
#define tamanioL1 80 //kb
#define tamanioL2 1280 //kb
#define tamanioL3 25000 //kb

//Matrices

int** randomMatrix(int n);

void freeMatrix(int ** a, int n);

int* randomArray(int n);

void shuffleArray(int* array, int n);

int* sequentialArray(int n);

struct cacheInicial cacheInicial(int *pInt, void *pVoid, void *pVoid1);

struct cacheInicial {
    int* L1;
    int* L2;
    int* L3;
};

//void setCPU();

#endif //PRACTICO1_UTILS_H
