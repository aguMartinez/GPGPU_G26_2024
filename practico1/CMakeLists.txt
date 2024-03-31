cmake_minimum_required(VERSION 3.27)
project(practico1 C)

set(CMAKE_C_STANDARD 11)

add_executable(practico1 main.c
        utils.c
        utils.h
        src/ej1PrefetchExplicito.c
        include/ej1PrefetchExplicito.h)
