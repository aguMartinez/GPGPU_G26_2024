#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

//MS(a = randomMatrix(i), interval);


/*  Linea de cache 64 bytes (16 integer)
 *
 *
 *
 */

int* llenarL1(){
    int n = tamanioL1 * 1000 / sizeof(int);
    int * a = randomArray(n);
    return a;
}



int* llenarL2(){
    int n = (tamanioL1 + tamanioL2) * 1000 / sizeof(int);
    int * a = randomArray(n);
    return a;
}

int* llenarL3(){
    int n = (tamanioL3 + tamanioL2 + tamanioL1)* 1000 / sizeof(int);
    int * a = randomArray(n);
    return a;
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


    int n = (tamanioL1 * 1000 / sizeof(int)) / 2; // Divido por 2 para que no se pase de la cache
    double intervals = 0;
    int cont = 0;

    for (int i = 0; i < 15; i++) {

        int* a = llenarL1();
        int suma;

        NS(suma = sumaArregloSecuencial(a, n), interval)

        printf("Suma de los elementos del arreglo saltando de a 16: %d\n", suma);

        printf("Tiempo de ejecucion de la suma secuencial de L1: %f ns\n", interval/n);

        cont = cont + 1;

        printf("cont: %d\n", cont);

        intervals = interval/n + intervals;

        free(a);
    }

    printf("Tiempo de ejecucion promedio de la suma secuencial de L1: %f ns\n", intervals/15);

}