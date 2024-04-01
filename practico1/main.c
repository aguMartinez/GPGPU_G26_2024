#include "include/cacheUtils.h"
#include "include/ej1TiempoPromedioAccesoCache.h"
#include "include/ej1PrefetchExplicito.h"
#include "include/ej2ReordenamientoYBlocking.h"
#include "include/cacheUtils.h"
#include "include/utils.h"

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