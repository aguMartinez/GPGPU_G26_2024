//
// Created by Valen on 26/3/2024.
//

#ifndef PRACTICO1_CACHEUTILS_H
#define PRACTICO1_CACHEUTILS_H
struct cacheInicial {
    int* L1;
    int* L2;
    int* L3;
};
#endif //PRACTICO1_CACHEUTILS_H


void liberarCache(struct cacheInicial c);

struct cacheInicial llenarL1(char shuffle);

struct cacheInicial llenarL2(char shuffle);

struct cacheInicial llenarL3(char shuffle);

struct cacheInicial cacheInicial(int *pInt, void *pVoid, void *pVoid1);

