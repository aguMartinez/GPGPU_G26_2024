#include "cacheUtils.h"
#include "ejercicio1.c"


int main() {

    char numEj;

    printf("Ingrese el numero de ejercicio a ejecutar: ");
    scanf("%c",&numEj);

    switch(numEj){
        case '1':
            printf("Ejercicio 1\n");
            ejercicio1();
            break;
        case '2':
            //ejercicio2();
            break;
        case '3':
            //ejercicio3();
            break;
    }
    return 1;
}