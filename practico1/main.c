#include "include/cacheUtils.h"
#include "include/ejercicio1.h"
#include "include/ej1PrefetchExplicito.h"
#include "include/ejercicio2.h"
#include <stdio.h>
#include "include/cacheUtils.h"


int main() {

    char numEj, parteEj;

    printf("Ingrese el numero de ejercicio a ejecutar: ");
    scanf("%c",&numEj);

    switch(numEj){
        case '1':
            printf("Ingrese parte de ejercicio a ejecutar: ");
            scanf(" %c",&parteEj);
            if(parteEj=='a')
                    ejercicio1();
            if(parteEj=='b')
                    ej1PrefetchExplicito();
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