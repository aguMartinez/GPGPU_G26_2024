
//Utilidades
#include "include/utils.h"
#include "include/cacheUtils.h"

//Ejercicio 1:
#include "include/ej1TiempoPromedioAccesoCache.h"
#include "include/ej1PrefetchExplicito.h"
#include "include/ej1AccesoDesalineado.h"
#include "include/cacheUtils.h"

//Ejercicio 2:
#include "include/ej2Reordenamiento.h"
#include "include/ej2Blocking.h"
#include "include/ej2CompetenciaSetCache.h"

//Bibliotecas estandar:
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    set_cpu_affinity(0);

    char numEj, parteEj;
    srand(time(NULL));

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
            if(parteEj=='c')
                    ej1AccesoDesalineado();
            break;

        case '2':
            printf("Ingrese parte de ejercicio a ejecutar: ");
            scanf(" %c",&parteEj);
            if(parteEj=='a')
                ej2Reordenamiento();
            if(parteEj=='b')
                ej2Blocking();
            if(parteEj=='c')
                ej2CompetenciaSetCache();
            break;
        case '3':
            ejercicio1();
            ej2Reordenamiento();
            break;
    }
    return 1;
}