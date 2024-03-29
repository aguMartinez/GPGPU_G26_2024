#include "../include/utils.h"
#include "../include/cacheUtils.h"
#include <stdio.h>
#include <malloc.h>


struct cacheInicial cacheInicial(int* pInt, int* pVoid, int* pVoid1) {
    struct cacheInicial result;
    result.L1 = pInt;
    result.L2 = pVoid;
    result.L3 = pVoid1;
    return result;
}


void liberarCache(struct cacheInicial c){
    free(c.L1);
    free(c.L2);
    free(c.L3);
}

/*
arrayElements:
    random: r
    sequential: s
*/
struct cacheInicial llenarL1(char shuffle){
    int n = (tamanioL1 * kb / sizeof(int))/2; //Ocupamos la mitad de la cache para que no haya misses

    int * a = sequentialArray(n);

    if (shuffle == 's') {
        shuffleArray(a, n);
    }

    return cacheInicial(a, NULL, NULL);
}



struct cacheInicial llenarL2(char shuffle){
    int n = tamanioL2 * kb / sizeof(int);
    int * a = sequentialArray(n);

    if (shuffle == 's') {
        shuffleArray(a, n);
    }

    n = tamanioL1 * kb / sizeof(int);
    int * b = sequentialArray(n);

    return cacheInicial(b, a, NULL);
}

struct cacheInicial llenarL3(char shuffle){
    int n = (tamanioL3)* kb / sizeof(int);

    int * a = sequentialArray(n);

    if (shuffle == 's') {
        shuffleArray(a, n);
    }

    n = tamanioL2 * kb / sizeof(int);
    int * b = sequentialArray(n);

    n = tamanioL1 * kb / sizeof(int);
    int * c = sequentialArray(n);

    return cacheInicial(c, b, a);
}


