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



void tareaParaArray(int* a, int n) {

    for (int i = 0; i < n; i++) {
        a[i] = a[i] + 2;
    }
}


int main() {

    if (freopen("output.txt", "w", stdout) == NULL) {
        perror("freopen failed");
        return EXIT_FAILURE;
    }

    int n = (80 + 1280 + 25000) * 1000 / sizeof(int);
    printf("n: %d\n", n);
    int * a = randomArray(n);

    for (int i = 0; i < n; i++) {
        MS(tareaParaArray(a, n), interval);
        printf("%d,%f\n", i, interval);
    }

    free(a);
}