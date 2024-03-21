#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include "utils.h"

//

int main() {
    setCPU();
    int n = 5000;
    int **a;

    for (int i = 1; i <= n; i++) {

        MS(a = createMatrix(i), interval);
            freeMatrix(a, i);
            printf("%d,%f  \n", i, interval);
    }
}