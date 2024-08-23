#include <stdio.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

void printVector(float* V, int n){
    for (int i = 0; i < n; i++){
        printf("%.1f ", V[i]);
    }
    printf("\n");
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
        
        // Guardar arreglo de vecinos:
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

struct GetWindows2 {
    unsigned int * windows;
    const float* img_in;
    const int width;
    const int height;
    const int W;
    const int radius;
    const int window_size;

    __device__
    void operator()(int id) {
        int row = id / width; // y
        int col = id % width; // x

        // Guardar arreglo de vecinos:
        for (int i = -radius; i <= radius; i++) { // dy
            for (int j = -radius; j <= radius; j++) { // dx
                int nCol = col + j; // nx
                int nRow = row + i; // ny
                
                if (nCol >= 0 && nCol < width && nRow >= 0 && nRow < height)
                    windows[(id * W * W) + ((i + radius) * W) + (j + radius)] = img_in[nRow * width + nCol] + (256 * id);
                else 
                    windows[(id * W * W) + ((i + radius) * W) + (j + radius)] = 256 * id;
            }
        }
    }
};

struct GetMedian {
    float* windows;
    const int window_size;

    __device__
    void insertion_sort(float* data, int size) {
        for (int i = 1; i < size; ++i) {
            float key = data[i];
            int j = i - 1;

            while (j >= 0 && data[j] > key) {
                data[j + 1] = data[j];
                j = j - 1;
            }
            data[j + 1] = key;
        }
    }

    __device__
    float operator()(int id) {
        int start = id * window_size;
        int end = id * window_size + window_size - 1;

        insertion_sort(windows + start, window_size);

        return windows[start + window_size / 2];
    }
};


struct GetMedian2 {
    unsigned int* windows;
    const int window_size;

    __device__
    unsigned int operator()(int id) {
        int offset = id * window_size;

        unsigned int median = windows[offset + window_size / 2] - (id * 256);

        return median;
    }
};

void filtro_mediana_gpu(float* img_in, float* img_out, int width, int height, int W) {
    int img_size = height * width;
    int radius = W / 2;
    int window_size = W * W;

    thrust::device_vector<unsigned int> d_windows(img_size * window_size);
    thrust::device_vector<float> d_img_in(img_in, img_in + img_size);

    thrust::for_each(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(img_size),
        GetWindows2{
            thrust::raw_pointer_cast(d_windows.data()),
            thrust::raw_pointer_cast(d_img_in.data()),
            width, height, W, radius, window_size
        }
    );

    thrust::sort(thrust::device, d_windows.begin(), d_windows.end());

    thrust::device_vector<unsigned int> d_medians(img_size);
    GetMedian2 getMedian2 = {thrust::raw_pointer_cast(d_windows.data()), window_size};

    thrust::transform(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(img_size),
        d_medians.begin(),
        getMedian2
    );

    thrust::copy(d_medians.begin(), d_medians.end(), img_out);

}

void filtro_mediana_gpu1(float* img_in, float* img_out, int width, int height, int W) {

    // Paso 1: Generar ventanas en paralelo
    int img_size = height * width;
    int radius = W / 2;

    printf("Img Size: %d\n", img_size);
    printf("W: %d\n", W);

    thrust::device_vector<float> d_windows(img_size * (W * W));
    thrust::device_vector<float> d_img_in(img_in, img_in + img_size);

    thrust::for_each(
        thrust::device, 
        thrust::counting_iterator<int>(0), 
        thrust::counting_iterator<int>(img_size), 
        GetWindows{
            thrust::raw_pointer_cast(d_windows.data()), 
            thrust::raw_pointer_cast(d_img_in.data()), 
            width, height, W, radius
        }
    );

    // Paso 2: Obtener medianas
    thrust::device_vector<float> d_medians(img_size);
    GetMedian getMedian = {thrust::raw_pointer_cast(d_windows.data()), W * W};

    thrust::transform(
        thrust::device, 
        thrust::counting_iterator<int>(0), 
        thrust::counting_iterator<int>(img_size), 
        d_medians.begin(), 
        getMedian
    );

    thrust::copy(d_medians.begin(), d_medians.end(), img_out);

}
