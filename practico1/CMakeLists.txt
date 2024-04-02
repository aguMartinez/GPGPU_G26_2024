cmake_minimum_required(VERSION 3.27)
project(practico1 C)

set(CMAKE_C_STANDARD 11)

add_executable(practico1 main.c
        include/cacheUtils.h
        include/ej1PrefetchExplicito.h
        include/ej1TiempoPromedioAccesoCache.h
        include/ej2Reordenamiento.h
        include/utils.h

        src/cacheUtils.c
        src/ej1PrefetchExplicito.c
        src/ej1TiempoPromedioAccesoCache.c
        src/ej1AccesoDesalineado.c
        src/ej2Reordenamiento.c
        src/ej2Blocking.c
        src/utils.c
        src/ej1AccesoDesalineado.c
        include/ej1AccesoDesalineado.h
        include/ej2Blocking.h
)
