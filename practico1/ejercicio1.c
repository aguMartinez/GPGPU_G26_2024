#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"
#include "cacheUtils.h"

int sumaArregloSecuencial(int* a, int n){

    int suma = 0;
    for (int i = 0; i < n; i++) {
        suma += a[i];
    }

    return suma;
}

int sumaArregloSaltos(int* a, int n, int paso){

    int suma = 0;
    for (int j = 0; j < paso; j++) {
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

/*
 * Tipos de recorridas posibles:
 * - Secuencial (s)
 * - Hops (h)
 * - Aleatoria (r)
 * */
void medirVelocidadL1(char tipoRecorrida) {


    switch (tipoRecorrida){
        case 's':
            if (freopen("output_ej1_L1_secuencial.csv", "w", stdout) == NULL) {
                perror("freopen failed");
            }
            break;

        case 'h':
            if (freopen("output_ej1_L1_salto.csv", "w", stdout) == NULL) {
                perror("freopen failed");
            }
                break;

        case 'r':
            if (freopen("output_ej1_L1_random.csv", "w", stdout) == NULL) {
                perror("freopen failed");
            }
            break;
    }

    int n = ((tamanioL1) * kb / sizeof(int)) ;
    struct cacheInicial c;
    int suma;
    double interval = 0;

    printf("Iteracion,Tiempo (ns),Suma\n");

    for (int i = 0; i < 100; i++) {

        if (tipoRecorrida == 'r') {
            c = llenarL1('s');
        } else {
            c = llenarL1('n');
        }

        int * b = c.L1;

        switch (tipoRecorrida) {
            case 'r':
                NS(suma = sumaArregloSecuencial(b, n), interval)
                break;
            case 'h':
                NS(suma = sumaArregloSaltos(b, n, 16), interval)
                break;
            case 's':
                NS(suma = sumaArregloSecuencial(b, n), interval)
                break;

        }

        printf("%2d,%10.2f,%10d\n", i+1,interval/n,suma);


        liberarCache(c);
    }

    setConsoleAsStdOutput();
}

void medirVelocidadL2(char tipoRecorrida) {

        switch (tipoRecorrida){
            case 's':
                if (freopen("output_ej1_L2_secuencial.csv", "w", stdout) == NULL) {
                    perror("freopen failed");
                }
                break;

            case 'h':
                if (freopen("output_ej1_L2_salto.csv", "w", stdout) == NULL) {
                    perror("freopen failed");
                }
                break;

            case 'r':
                if (freopen("output_ej1_L2_random.csv", "w", stdout) == NULL) {
                    perror("freopen failed");
                }
                break;
        }

        int n = ((tamanioL2) * kb / sizeof(int)) ;
        struct cacheInicial c;
        int suma;
        double interval;

        printf("Iteracion,Tiempo (ns),Suma\n");

        for (int i = 0; i < 100; i++) {

            if (tipoRecorrida == 'r') {
                c = llenarL2('s');
            } else {
                c = llenarL2('n');
            }

            int * b = c.L2;

            switch (tipoRecorrida) {
                case 'r':
                    NS(suma = sumaArregloSecuencial(b, n), interval)
                    break;
                case 'h':
                    NS(suma = sumaArregloSaltos(b, n, 16), interval)
                    break;
                case 's':
                    NS(suma = sumaArregloSecuencial(b, n), interval)
                    break;

            }

            printf("%2d,%10.2f,%10d\n", i+1,interval/n,suma);
            liberarCache(c);
        }
    setConsoleAsStdOutput();

}

void medirVelocidadL3(char tipoRecorrida) {

    switch (tipoRecorrida){
        case 's':
            if (freopen("output_ej1_L3_secuencial.csv", "w", stdout) == NULL) {
                perror("freopen failed");
            }
            break;

        case 'h':
            if (freopen("output_ej1_L3_salto.csv", "w", stdout) == NULL) {
                perror("freopen failed");
            }
            break;

        case 'r':
            if (freopen("output_ej1_L3_random.csv", "w", stdout) == NULL) {
                perror("freopen failed");
            }
            break;
    }

    int n = ((tamanioL3) * kb / sizeof(int)) ;
    struct cacheInicial c;
    int suma;
    double interval;

    printf("Iteracion,Tiempo (ns),Suma\n");

    for (int i = 0; i < 100; i++) {

        if (tipoRecorrida == 'r') {
            c = llenarL3('s');
        } else {
            c = llenarL3('n');
        }

        int * b = c.L3;

        switch (tipoRecorrida) {
            case 'r':
                NS(suma = sumaArregloSecuencial(b, n), interval)
                break;
            case 'h':
                NS(suma = sumaArregloSaltos(b, n, 16), interval)
                break;
            case 's':
                NS(suma = sumaArregloSecuencial(b, n), interval)
                break;

        }

        printf("%2d,%10.2f,%10d\n", i+1,interval/n,suma);
        liberarCache(c);
    }
    setConsoleAsStdOutput();
}



void ejercicio1() {

    printf("==============================\n");
    printf("Medir velocidad de acceso a memoria cache\n");
    printf("==============================\n");

    printf("Velocidad de acceso a cache L1\n");
    printf("==============================\n");
    printf("| Secuencia | Saltos | Aleatorio |\n");
    medirVelocidadL1('s');
    medirVelocidadL1('h');
    medirVelocidadL1('r');
    printf("==============================\n");

    printf("Velocidad de acceso a cache L2\n");
    printf("==============================\n");
    printf("| Secuencia | Saltos | Aleatorio |\n");
    medirVelocidadL2('s');
    medirVelocidadL2('h');
    medirVelocidadL2('r');
    printf("==============================\n");

    printf("Velocidad de acceso a cache L3\n");
    printf("==============================\n");
    printf("| Secuencia | Saltos | Aleatorio |\n");
    medirVelocidadL3('s');
    medirVelocidadL3('h');
    medirVelocidadL3('r');
    printf("==============================\n");

}
