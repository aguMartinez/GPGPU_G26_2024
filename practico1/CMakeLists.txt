cmake_minimum_required(VERSION 3.27)
project(practico1 C)

set(CMAKE_C_STANDARD 11)

add_executable(practico1 main.c
        include/cacheUtils.h
        include/ej1PrefetchExplicito.h
        include/ejercicio1.h
        include/ejercicio2.h
        include/utils.h

        src/cacheUtils.c
        src/ej1PrefetchExplicito.c
        src/ejercicio1.c
        src/ejercicio2.c
        src/utils.c
)
