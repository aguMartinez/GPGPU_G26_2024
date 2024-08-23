#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <vector>
#include <algorithm>
#include "CImg.h"

#include <iostream>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

using namespace cimg_library;

void filtro_mediana_gpu(float * img_in, float * img_out, int width, int height, int W);
void filtro_mediana_cpu(float * img_in, float * img_out, int width, int height, int W);

int main(int argc, char** argv){
    
    const char * path;

    if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
    else
        path = argv[argc-1];

    CImg<float> image(path);
    
    // Crear dos imágenes de salida separadas
    CImg<float> image_out_cpu(image.width(), image.height(),1,1,0);
    CImg<float> image_out_gpu(image.width(), image.height(),1,1,0);

    float *img_matrix = image.data();
    float *img_out_matrix_cpu = image_out_cpu.data();
    float *img_out_matrix_gpu = image_out_gpu.data();

    // [begin] Evento de tiempo para CPU
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);

    cudaEventRecord(start_cpu, 0);

    filtro_mediana_cpu(img_matrix, img_out_matrix_cpu, image.width(), image.height(), 3);

    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);

    float elapsed_cpu = 0;
    cudaEventElapsedTime(&elapsed_cpu, start_cpu, stop_cpu);

    printf("Tiempo CPU: %f ms\n", elapsed_cpu);
    // [end] Evento de tiempo para CPU

    image_out_cpu.save("output_cpu.pgm");

    // [begin] Evento de tiempo para GPU
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu, 0);

    filtro_mediana_gpu(img_matrix, img_out_matrix_gpu, image.width(), image.height(), 3);

    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);

    float elapsed_gpu = 0;
    cudaEventElapsedTime(&elapsed_gpu, start_gpu, stop_gpu);

    printf("Tiempo GPU: %f ms\n", elapsed_gpu);
    // [end] Evento de tiempo para GPU

    // Imprimir las primeras entradas del resultado GPU
    for (int i = 0; i < image.width() * image.height(); i++) {
        if (img_out_matrix_gpu[i] != img_out_matrix_cpu[i]) {
            printf("Elemento cpu: %f, elemento gpu: %f, i: %d \n", img_out_matrix_cpu[i], img_out_matrix_gpu[i], i);
        }
    }
    printf("\n");

    image_out_gpu.save("output_gpu.pgm");
       
    return 0;
}
