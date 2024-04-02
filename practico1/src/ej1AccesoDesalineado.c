#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/utils.h"

#define TAMANIO_ARREGLO 10000


void accesoAlineado(int* arreglo, int tamanio) {
    volatile int suma = 0;
    for (int i = 0; i < tamanio; i++) {
        int valor = *((int*)(arreglo + i));
        suma += valor;
    }
}

void accesoSimuladoDesalineado(char* arreglo, int tamanio) {
    volatile int suma = 0;
    for (int i = 1; i < tamanio; i+=sizeof(int)) {
        int valor = *((int*)(arreglo + i));
        suma += valor;
    }
}

int ej1AccesoDesalineado() {

    if (freopen("output_ej1_acceso_desalineado.csv", "w", stdout) == NULL) {
        perror("freopen failed");
    }

    char* arregloDesalineado;
    int* arregloAlineado, n_alineado, n_desalineado;

    printf("Tamaño arreglo,Tiempo acceso alineado,Tiempo acceso desalineado\n");


    for(int i = 0; i < 1000; i++){
        n_alineado = TAMANIO_ARREGLO * i * sizeof(int);
        n_desalineado = TAMANIO_ARREGLO * i * sizeof(int) + sizeof(int) - 1;
        arregloAlineado = (int*)_aligned_malloc(n_alineado, 64);
        arregloDesalineado = (char*)_aligned_malloc(n_desalineado, 64);

        double tiempoAlineado, tiempoDesalineado;

        NS(accesoSimuladoDesalineado(arregloDesalineado,  n_desalineado), tiempoDesalineado);
        NS(accesoAlineado(arregloAlineado, n_alineado / sizeof(int)), tiempoAlineado);

        printf("%d,%f,%f\n", TAMANIO_ARREGLO * i, tiempoAlineado, tiempoDesalineado);

        _aligned_free(arregloAlineado);
        _aligned_free(arregloDesalineado);
    }

    return 0;
}

















