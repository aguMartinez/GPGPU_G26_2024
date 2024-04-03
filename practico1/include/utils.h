//
// Created by amartinez on 20/3/2024.
//

#ifndef PRACTICO1_UTILS_H
#define PRACTICO1_UTILS_H

//1 kb = 1000b
#define kb 1000

//1 mb = 1000b
#define MB 1000000

//PCores
#define tamanioL1P 80 //kb (p-core)
#define tamanioL2P 1280 //kb (p-core)

//ECores

#define tamanioL1 32 //kb (e-core)
#define tamanioL2 2000 //kb (e-core)

//Cache L3 Shared
#define tamanioL3 25000 //kb


#define MS(f,elap)                                                                                           \
        {                                                                                                    \
        struct timespec t_ini,t_fin;                                                                         \
            clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                          \
            f;                                                                                               \
            clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                          \
            elap = 1000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec)/1000000.0;         \
        }

#define NS(f,elap)                                                                                           \
        {                                                                                                    \
        struct timespec t_ini,t_fin;                                                                         \
            clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                          \
            f;                                                                                               \
            clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                          \
            elap = 1000000000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec);             \
        }


//Utilidades de Matrices

/* Crea y devuelve una matriz de tamaño nxn sin cargar (con basura)*/
int** initializeMatrix(int n);

/* Crea y devuelve una matriz secuencial, A[i][j]== i*10+j */
int** sequentialMatrix(int n);

/* Crea y devuelve una matriz de tamaño nxn cargada con numeros aleatorios*/
int** randomMatrix(int n);

/* Crea y devuelve una matriz de tamaño nxn con los mismos elementos que A (no comparte memoria)*/
int** cloneMatrix(int** A, int n);

/*Imprime por la salida estándar la matriz A (que es de tamaño nxn)*/
void printMatrix(int** A, int n);

/*Libera la memoria de la matriz A (que es de tamaño nxn)*/
void freeMatrix(int ** a, int n);

//Utilidades de arreglos

int* randomArray(int n);

void shuffleArray(int* array, int n);

int* sequentialArray(int n);

void setConsoleAsStdOutput();
void set_cpu_affinity(int cpu_id);
int** sequentialAlignedMatrix(int n);
void freeAlignedMatrix(int ** a, int n);
//void setCPU();

#endif //PRACTICO1_UTILS_H
