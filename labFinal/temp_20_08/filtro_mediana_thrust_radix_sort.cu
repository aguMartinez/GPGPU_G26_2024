#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#define WINDOW_SIZE 3


__device__
int float_to_ordered_int(float value) {
    return static_cast<int>(value);
}

__device__
void split(float* input, float* output, int* temp, int window_size, int bit) {
    int mask = 1 << bit;

    // Paso 1: Crear el arreglo temporal e para almacenar el bit_value
    for (int i = 0; i < window_size; i++) {
        int int_representation = float_to_ordered_int(input[i]);
        int bit_value = (int_representation & mask);

        bit_value = bit_value >> bit;

        temp[i] = (bit_value == 0) ? 1 : 0;

    }

    int ult_valor_temp = temp[window_size - 1];

    thrust::exclusive_scan(thrust::seq, temp, temp + window_size, temp);

    // Calcular cuántos elementos tienen bit_value = 0
    int totalFalses = temp[window_size - 1] + ult_valor_temp;


    // Paso 3: Asignar posiciones finales basadas en los bits
    for (int i = 0; i < window_size; i++) {
        int index;
        if ((float_to_ordered_int(input[i]) & mask) >> bit) {
            index = i - temp[i] + totalFalses;
        } else {
            index = temp[i];
        }

        // Verificar que el índice esté dentro del rango
        if (index >= 0 && index < window_size) {
            output[index] = input[i];
        }
    }

    // Paso 4: Copiar los resultados de vuelta a input
    for (int i = 0; i < window_size; i++) {
        input[i] = output[i];
    }
}


// Implementación de Radix Sort usando Split basado en bits
__device__
void radix_sort(float* window, int window_size) {
    float output[WINDOW_SIZE * WINDOW_SIZE];
    int temp[WINDOW_SIZE * WINDOW_SIZE];

    for (int bit = 0; bit < 8; bit++) {
        split(window, output, temp, window_size, bit);
    }
}

struct GetWindows {
    float* windows;
    const float* img_in;
    const int width;
    const int height;
    const int W;
    const int radius;

    __device__
    void operator()(int id) {
        if (id >= width * height) return;

        int row = id / width; // y
        int col = id % width; // x
        
        for (int i = -radius; i <= radius; i++) { // dy
            for (int j = -radius; j <= radius; j++) { // dx
                int nCol = col + j; // nx
                int nRow = row + i; // ny
                
                if (nCol >= 0 && nCol < width && nRow >= 0 && nRow < height)
                    windows[(id * W * W) + ((i + radius) * W) + (j + radius)] = img_in[nRow * width + nCol];
                else 
                    windows[(id * W * W) + ((i + radius) * W) + (j + radius)] = 0.0f;
            }
        }
    }
};

struct GetMedian {
    float* windows;
    const int window_size;

    __device__
    float operator()(int id) {
        int start = id * window_size;

        radix_sort(windows + start, window_size);

        return windows[start + window_size / 2];
    }
};

void filtro_mediana_gpu(float* img_in, float* img_out, int width, int height, int W) {

    int img_size = height * width;
    int radius = W / 2;

    thrust::device_vector<float> d_windows(img_size * (W * W));
    thrust::device_vector<float> d_img_in(img_in, img_in + img_size);

    thrust::for_each(
        thrust::counting_iterator<int>(0), 
        thrust::counting_iterator<int>(img_size), 
        GetWindows{
            thrust::raw_pointer_cast(d_windows.data()), 
            thrust::raw_pointer_cast(d_img_in.data()), 
            width, height, W, radius
        }
    );

    thrust::device_vector<float> d_medians(img_size);
    GetMedian getMedian = {thrust::raw_pointer_cast(d_windows.data()), W * W};

    thrust::transform(
        thrust::counting_iterator<int>(0), 
        thrust::counting_iterator<int>(img_size), 
        d_medians.begin(), 
        getMedian
    );

    thrust::copy(d_medians.begin(), d_medians.end(), img_out);
}
