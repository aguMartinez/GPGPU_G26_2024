#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

//MS(a = randomMatrix(i), interval);


struct cacheInicial llenarL1(){
    int n = tamanioL1 * kb / sizeof(int);
    int * a = sequentialArray(n);
    return cacheInicial(a, NULL, NULL);
}



struct cacheInicial llenarL2(){
    int n = tamanioL2 * kb / sizeof(int);
    int * a = sequentialArray(n);

    n = tamanioL1 * kb / sizeof(int);
    int * b = sequentialArray(n);

    return cacheInicial(b, a, NULL);
}

struct cacheInicial llenarL3(){
    int n = (tamanioL3)* kb / sizeof(int);
    int * a = sequentialArray(n);

    n = tamanioL2 * kb / sizeof(int);
    int * b = sequentialArray(n);

    n = tamanioL1 * kb / sizeof(int);
    int * c = sequentialArray(n);

    return cacheInicial(c, b, a);
}

int sumaArregloSecuencial(int* a, int n){

    int suma = 0;

    for (int i = 0; i < n; i++) {
        suma += a[i];
    }

    return suma;
}

int sumaArregloSaltando(int* a, int n){

    int suma = 0;
    for (int j = 0; j < 16; j++) {
        for (int i = j; i < n; i+= 16) {
            suma += a[i];
        }
    }

    return suma;
}


int sumaArregloRandom(int* a, int n) {

    int suma = 0;
    for (int i = 0; i < n; i++) {
        suma += a[a[i]];
    }
    return suma;
}


int main() {

    int seed = time(NULL);
    srand(seed);

   /* if (freopen("output.txt", "w", stdout) == NULL) {
        perror("freopen failed");
        return EXIT_FAILURE;
    }*/

    int n = ((tamanioL3) * kb / sizeof(int)) ; // Divido por 2 para que no se pase de la cacheInicial
    double intervals = 0;
    int cont = 0;

    for (int i = 0; i < 100; i++) {

        struct cacheInicial c = llenarL3();
        int * b = c.L3;
        shuffleArray(b, sizeof(b)/sizeof(int));
        int suma;

        NS(suma = sumaArregloRandom(b, n), interval)

        printf("Suma de los elementos del arreglo saltando de a 16: %d\n", suma);

        printf("Tiempo de ejecucion de la suma secuencial de L1: %f ns\n", interval/n);

        cont = cont + 1;

        printf("cont: %d\n", cont);

        intervals = interval/n + intervals;

        free(b);
    }

    printf("Tiempo de ejecucion promedio de la suma secuencial de L1: %f ns\n", intervals/100);

}