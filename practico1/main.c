#include "include/cacheUtils.h"
#include "include/ejercicio1.h"
#include "include/ejercicio2.h"
#include <stdio.h>
#include "include/cacheUtils.h"


int main() {

    char numEj;

    printf("Ingrese el numero de ejercicio a ejecutar: ");
    scanf("%c",&numEj);

    switch(numEj){
        case '1':
            ejercicio1();
            break;
        case '2':
            ejercicio2();
            break;
        case '3':
            ejercicio1();
            ejercicio2();
            break;
    }
    return 1;
}