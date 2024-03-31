//
// Created by agu on 31/3/2024.
//

#include "../include/utils.h"

void busquedaBinariaSinPrefetch(int* a, int cantElems, int clave){
    int bajo = 0;
    int alto = cantElems-1;
    int medio;

    while(bajo <= alto){
        medio = (alto-bajo)/2;
        if(a[medio] < clave)
            bajo = medio+1;
        else if a[medio] == clave
            return medio;
        else
            alto = mid-1;
    }
    return -1;
}

void busquedaBinariaPrefetch(int* a, int cantElems, int clave){
    int bajo = 0,
    alto = cantElems-1,
    medio;

    while(bajo <= alto){
        medio = (alto-bajo)/2;
        __builtin_prefetch (&a[(medio + 1 + alto)/2], 0, 1);
        __builtin_prefetch (&a[(bajo + medio - 1)/2], 0, 1);

        if(a[medio] < clave)
            bajo = medio+1;
        else if a[medio] == clave
        return medio;
        else
        alto = mid-1;
    }

    return -1;
}

void ej1PrefetchExplicito(){
    int* a = sequentialArray();
    int res;

    MS(res = busquedaBinariaSinPrefetch(a);)
    MS(res = busquedaBinariaPrefetch(a);)
}