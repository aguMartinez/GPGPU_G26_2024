//
// Created by agu on 31/3/2024.
//

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "../include/utils.h"
#include "../include/cacheUtils.h"

int busquedaBinariaSinPrefetch(int* a, int cantElems, int clave){
    int bajo = 0;
    int alto = cantElems-1;
    int medio;

    while(bajo <= alto){
        medio = (alto-bajo)/2;
        if(a[medio] < clave)
            bajo = medio+1;
        else if (a[medio] == clave)
            return medio;
        else
            alto = medio-1;
    }
    return -1;
}

int busquedaBinariaPrefetch(int* a, int cantElems, int clave){
    int bajo = 0,
    alto = cantElems-1,
    medio;

    while(bajo <= alto){
        medio = (alto-bajo)/2;
        __builtin_prefetch (&a[(medio + 1 + alto)/2], 0, 1);
        __builtin_prefetch (&a[(bajo + medio - 1)/2], 0, 1);

        if(a[medio] < clave)
            bajo = medio+1;
        else if (a[medio] == clave)
            return medio;
        else
        alto = medio-1;
    }

    return -1;
}

void ej1PrefetchExplicito(){
    int size = 1*MB; //Arreglo de 2GB

    int elementos = size/sizeof(int);

    int* a = sequentialArray(size/elementos);
    int clave = rand() % elementos;
    int res;

    double tiempoSinPrefetch = 0;
    double tiempoPrefetch = 0;

    NS(res = busquedaBinariaSinPrefetch(a, elementos, clave), tiempoSinPrefetch)
    NS(res = busquedaBinariaPrefetch(a, elementos, clave), tiempoPrefetch)

    printf("Encontrado: %d",res);
    printf("Tiempo sin prefetch: %d", tiempoSinPrefetch);
    printf("Tiempo con prefetch: %d", tiempoPrefetch);
}