#include <stdio.h>
#include <time.h>
#include "../include/utils.h"
#include "../include/cacheUtils.h"

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

    int n = ((tamanioL1) * kb / sizeof(int))/2 ;
    struct cacheInicial c;
    int suma;
    double interval = 0;

    printf("Iteracion,Tiempo (ns),Suma\n");

    for (int i = 0; i < 1000; i++) {

        int * b;

        switch (tipoRecorrida) {
            case 'r':
                c = llenarL1('s');
                b = c.L1;
                NS(suma = sumaArregloRandom(b, n), interval)
                break;
            case 'h':
                c = llenarL1('n');
                b = c.L1;
                NS(suma = sumaArregloSaltos(b, n, 16), interval)
                break;
            case 's':
                c = llenarL1('n');
                b = c.L1;
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

        for (int i = 0; i < 1000; i++) {

            int * b;

            switch (tipoRecorrida) {
                case 'r':
                    c = llenarL2('s');
                    b = c.L2;
                    NS(suma = sumaArregloRandom(b, n), interval)
                    break;
                case 'h':
                    c = llenarL2('n');
                    b = c.L2;
                    NS(suma = sumaArregloSaltos(b, n, 16), interval)
                    break;
                case 's':
                    c = llenarL2('n');
                    b = c.L2;
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

    for (int i = 0; i < 1000; i++) {

        int * b;

        switch (tipoRecorrida) {
            case 'r':
                c = llenarL3('s');
                b = c.L3;
                NS(suma = sumaArregloRandom(b, n), interval)
                break;
            case 'h':
                c = llenarL3('n');
                b = c.L3;
                NS(suma = sumaArregloSaltos(b, n, 16), interval)
                break;
            case 's':
                c = llenarL3('n');
                b = c.L3;
                NS(suma = sumaArregloSecuencial(b, n), interval)
                break;

        }

        printf("%2d,%10.2f,%10d\n", i+1,interval/n,suma);
        liberarCache(c);
    }
    setConsoleAsStdOutput();
}

/*

void accesoAMemoriaAlineado(int* a, int n) {

    if (freopen("output_ej1_acceso_alineado.csv", "w", stdout) == NULL) {
        perror("freopen failed");
    }

    int n = ((tamanioL3) * kb / sizeof(int)) ;
    struct cacheInicial c;
    int suma;
    double interval;

    printf("Iteracion,Tiempo (ns),Suma\n");

    for (int i = 0; i < 1000; i++) {

        int * b;

        switch (tipoRecorrida) {
            case 'r':
                c = llenarL3('s');
                b = c.L3;
                NS(suma = sumaArregloRandom(b, n), interval)
                break;
            case 'h':
                c = llenarL3('n');
                b = c.L3;
                NS(suma = sumaArregloSaltos(b, n, 16), interval)
                break;
            case 's':
                c = llenarL3('n');
                b = c.L3;
                NS(suma = sumaArregloSecuencial(b, n), interval)
                break;

        }

        printf("%2d,%10.2f,%10d\n", i+1,interval/n,suma);
        liberarCache(c);
    }
    setConsoleAsStdOutput();
}

void accesoAMemoriaDeslineado(int* a, int n) {

    int suma = 0;
    for (int i = 0; i < n; i+=15) {
        suma += a[i];
    }
}

*/

void ejercicio1() {
    medirVelocidadL1('h');
    medirVelocidadL1('r');
    medirVelocidadL1('s');

    medirVelocidadL2('s');
    medirVelocidadL2('h');
    medirVelocidadL2('r');

    medirVelocidadL3('s');
    medirVelocidadL3('h');
    medirVelocidadL3('r');

    printf("Ejercicio 1 done \n");
    printf("Los resultados se guardaron en archivos .csv en la carpeta del ejecutable \n");
}
