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


int** createMatrix(int n);

void freeMatrix(int ** a, int n);

void setCPU();

#endif //PRACTICO1_UTILS_H
