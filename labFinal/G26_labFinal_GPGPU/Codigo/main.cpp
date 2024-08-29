#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <vector>
#include <algorithm>
#include "CImg.h"

#include <iostream>
#include <nvToolsExt.h>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <math.h>


#define CUDA_EVENT_START(start, stop)       \
    cudaEventCreate(&start);                \
    cudaEventCreate(&stop);                 \
    cudaEventRecord(start, 0);

#define CUDA_EVENT_STOP(start, stop, elapsed_time)    \
    cudaEventRecord(stop, 0);                         \
    cudaEventSynchronize(stop);                       \
    cudaEventElapsedTime(&elapsed_time, start, stop); \
    cudaEventDestroy(start);                          \
    cudaEventDestroy(stop);



#define PROM_Y_DEVEST(f, start, stop, elapsed_time, datos, n, prom, devest) \
    prom = 0.0; \
    suma_cuadrados = 0.0; \
    for (int i = 0; i < n; i++) { \
        CUDA_EVENT_START(start, stop); \
        f; \
        CUDA_EVENT_STOP(start, stop, elapsed_time); \
        datos[i] = elapsed_time; \
        prom += elapsed_time; \
    } \
    prom /= n; \
    for (int i = 0; i < n; i++) { \
        suma_cuadrados += pow(datos[i] - prom, 2); \
    } \
    devest = sqrt(suma_cuadrados / (n - 1));

#ifndef _UCHAR
#define _UCHAR
    typedef unsigned char uchar;
#endif

using namespace cimg_library;

void filtro_mediana_cpu(uchar * img_in, uchar * img_out, int width, int height, int W);

// v1: SHARED
void filtro_mediana_gpu_baseline(uchar * img_in, uchar * img_out, int width, int height, int W, int blockSize);
void filtro_mediana_gpu_v1_0(uchar * img_in, uchar * img_out, int width, int height, int W, int blockSize);
void filtro_mediana_gpu_v1_1(uchar * img_in, uchar * img_out, int width, int height, int W, int blockSize);
void filtro_mediana_gpu_v1_2(uchar* img_in, uchar* img_out, int width, int height, int W, int blockSize);

// v2: THRUST
void filtro_mediana_gpu_v2_0(uchar* img_in, uchar* img_out, int width, int height, int W);
void filtro_mediana_gpu_v2_1(uchar* img_in, uchar* img_out, int width, int height, int W);

// v3: RADIX SORT
void filtro_mediana_gpu_v3(uchar* img_in, uchar* img_out, int width, int height, int W, int blockSize);

// v4: BUCKETS
void filtro_mediana_gpu_v4_0(uchar * img_in, uchar * img_out, int width, int height, int W, int blockSize);
void filtro_mediana_gpu_v4_1(uchar * img_in, uchar * img_out, int width, int height, int W, int blockSize);
void filtro_mediana_gpu_v4_2(uchar * img_in, uchar * img_out, int width, int height, int W, int blockSize);


int main(int argc, char** argv){

    char * path;

    if (argc < 4){
        printf("Debe ingresar el nombre del archivo y tamaño de ventana\n");
        return 1;
    }

    path = argv[1];
    printf("Path: %s\n", path);
    
    int W = atoi(argv[2]);
    printf("W: %d\n", W);

    int blockSize = atoi(argv[3]);
    printf("Block Size: %d x %d\n", blockSize, blockSize);

    CImg<uchar> image(path);

    printf("Tamaño de imagen: %d \n", image.width() * image.height());

    // Crear dos imágenes de salida separadas
    CImg<uchar> image_out_cpu(image.width(), image.height(),1,1,0);
    CImg<uchar> image_out_gpu(image.width(), image.height(),1,1,0);

    uchar *img_matrix = image.data();
    uchar *img_out_matrix_cpu = image_out_cpu.data();
    uchar *img_out_matrix_gpu = image_out_gpu.data();

    // Preparo variables para las pruebas
    int iters = 10;
    float elapsed_time, datos[iters], suma_cuadrados;
    cudaEvent_t start, stop;
    char out_path[50];


    // HEADER INFORME:

    printf("-----------------------------\n");
    printf("INFORME: \n");
    printf("-----------------------------\n");
    printf("Version,Prom (ms),Devest (ms)\n");
    printf("-----------------------------\n");

        


   
    
    // CPU: 
    float prom_cpu, devest_cpu;
    PROM_Y_DEVEST(
        filtro_mediana_cpu(img_matrix, img_out_matrix_cpu, image.width(), image.height(), W), 
        start, stop, elapsed_time, datos, iters, prom_cpu, devest_cpu
    )
    
    
    image_out_cpu.save("img_out/output_cpu.pgm");
    
    printf("CPU,%.2f,%.2f\n", prom_cpu, devest_cpu);
    
    
    
    // GPU Baseline:
    float prom_gpu_baseline, devest_gpu_baseline;
    
    PROM_Y_DEVEST(
        filtro_mediana_gpu_baseline(img_matrix, img_out_matrix_gpu, image.width(), image.height(), W, blockSize), 
        start, stop, elapsed_time, datos, iters, prom_gpu_baseline, devest_gpu_baseline
    )
    

    sprintf(out_path, "img_out/output_gpu_baseline_%d.pgm", W);
    image_out_gpu.save(out_path);

    printf("GPU Baseline,%.2f,%.2f\n", prom_gpu_baseline, devest_gpu_baseline);

    // GPU v1.0:
    float prom_gpu_1_0, devest_gpu_1_0;
    
    PROM_Y_DEVEST(
        filtro_mediana_gpu_v1_0(img_matrix, img_out_matrix_gpu, image.width(), image.height(), W, blockSize), 
        start, stop, elapsed_time, datos, iters, prom_gpu_1_0, devest_gpu_1_0
    )
    

    sprintf(out_path, "img_out/output_gpu_1_0_%d.pgm", W);
    image_out_gpu.save(out_path);

    printf("GPU v1.0,%.2f,%.2f\n", prom_gpu_1_0, devest_gpu_1_0);
    

    // GPU v1.1:
    float prom_gpu_1_1, devest_gpu_1_1;
    
    PROM_Y_DEVEST(
        filtro_mediana_gpu_v1_1(img_matrix, img_out_matrix_gpu, image.width(), image.height(), W, blockSize), 
        start, stop, elapsed_time, datos, iters, prom_gpu_1_1, devest_gpu_1_1
    )
    
    sprintf(out_path, "img_out/output_gpu_1_1_%d.pgm", W);

    image_out_gpu.save(out_path);
    printf("GPU v1.1,%.2f,%.2f\n", prom_gpu_1_1, devest_gpu_1_1);


    // GPU v1.2:
    float prom_gpu_1_2, devest_gpu_1_2;
    
    PROM_Y_DEVEST(
        filtro_mediana_gpu_v1_2(img_matrix, img_out_matrix_gpu, image.width(), image.height(), W, blockSize), 
        start, stop, elapsed_time, datos, iters, prom_gpu_1_2, devest_gpu_1_2
    )
    
    sprintf(out_path, "img_out/output_gpu_1_2_%d.pgm", W);

    image_out_gpu.save(out_path);
    printf("GPU v1.2,%.2f,%.2f\n", prom_gpu_1_2, devest_gpu_1_2);


    // GPU v2.0:
    float prom_gpu_2_0, devest_gpu_2_0;

    PROM_Y_DEVEST(
        filtro_mediana_gpu_v2_0(img_matrix, img_out_matrix_gpu, image.width(), image.height(), W), 
        start, stop, elapsed_time, datos, iters, prom_gpu_2_0, devest_gpu_2_0
    )


    sprintf(out_path, "img_out/output_gpu_2_0_%d.pgm", W);
    image_out_gpu.save(out_path);
   
    printf("GPU v2.0,%.2f,%.2f\n", prom_gpu_2_0, devest_gpu_2_0);


    // GPU v2.1:
    float prom_gpu_2_1, devest_gpu_2_1;

    PROM_Y_DEVEST(
        filtro_mediana_gpu_v2_1(img_matrix, img_out_matrix_gpu, image.width(), image.height(), W), 
        start, stop, elapsed_time, datos, iters, prom_gpu_2_1, devest_gpu_2_1
    )


    sprintf(out_path, "img_out/output_gpu_2_1_%d.pgm", W);
    image_out_gpu.save(out_path);
    
    printf("GPU v2.1,%.2f,%.2f\n", prom_gpu_2_1, devest_gpu_2_1);


    // GPU v3:
    float prom_gpu_3, devest_gpu_3;

    PROM_Y_DEVEST(
         filtro_mediana_gpu_v3(img_matrix, img_out_matrix_gpu, image.width(), image.height(), W, blockSize), 
         start, stop, elapsed_time, datos, iters, prom_gpu_3, devest_gpu_3
    )

    sprintf(out_path, "img_out/output_gpu_3_%d.pgm", W);

    image_out_gpu.save(out_path);
    printf("GPU v3,%.2f,%.2f\n", prom_gpu_3, devest_gpu_3);

    // GPU v4.0
    float prom_gpu_4_0, devest_gpu_4_0;
    int blockSize_4 = (blockSize < 12) ? blockSize : 12;
    
    PROM_Y_DEVEST(
        filtro_mediana_gpu_v4_0(img_matrix, img_out_matrix_gpu, image.width(), image.height(), W, blockSize_4), 
        start, stop, elapsed_time, datos, iters, prom_gpu_4_0, devest_gpu_4_0
    )

    sprintf(out_path, "img_out/output_gpu_4_0_%d.pgm", W);

    image_out_gpu.save(out_path);
    
    printf("GPU v4.0,%.2f,%.2f\n", prom_gpu_4_0, devest_gpu_4_0);

    // GPU v4.1
    float prom_gpu_4_1, devest_gpu_4_1;
    
    PROM_Y_DEVEST(
        filtro_mediana_gpu_v4_1(img_matrix, img_out_matrix_gpu, image.width(), image.height(), W, blockSize_4), 
        start, stop, elapsed_time, datos, iters, prom_gpu_4_1, devest_gpu_4_1
    )

    sprintf(out_path, "img_out/output_gpu_4_1_%d.pgm", W);
    image_out_gpu.save(out_path);
    
    printf("GPU v4.1,%.2f,%.2f\n", prom_gpu_4_1, devest_gpu_4_1);



    // GPU v4.2
   
    float prom_gpu_4_2, devest_gpu_4_2;
    
    PROM_Y_DEVEST(
        filtro_mediana_gpu_v4_2(img_matrix, img_out_matrix_gpu, image.width(), image.height(), W, blockSize_4), 
        start, stop, elapsed_time, datos, iters, prom_gpu_4_2, devest_gpu_4_2
    )

    sprintf(out_path, "img_out/output_gpu_4_2_%d.pgm", W);
    image_out_gpu.save(out_path);
    
    printf("GPU v4.2,%.2f,%.2f\n", prom_gpu_4_2, devest_gpu_4_2);

    printf("-----------------------------\n");

    printf("Chau!\n");


    return 0;
}
