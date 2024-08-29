#include <stdio.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#ifndef _UCHAR
#define _UCHAR
    typedef unsigned char uchar;
#endif

struct GetWindows2 {
    unsigned int * windows;
    const uchar* img_in;
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
        unsigned int base_offset = id * W * W;

        for (int i = -radius; i <= radius; i++) { // dy
            for (int j = -radius; j <= radius; j++) { // dx
                int nCol = col + j; // nx
                int nRow = row + i;
                unsigned int offset = base_offset + (i + radius) * W + (j + radius);

                if (nCol >= 0 && nCol < width && nRow >= 0 && nRow < height)
                    windows[offset] = static_cast<unsigned int>(img_in[nRow * width + nCol]) + (256 * id);
                else 
                    windows[offset] = (256 * id);
            }
        }
    }
};

struct GetMedian2 {
    unsigned int* windows;
    const int window_size;

    __device__
    unsigned int operator()(int id) {
        unsigned int offset = id * window_size;

        unsigned int median = windows[offset + window_size / 2] - (id * 256);

        return median;
    }
};

void filtro_mediana_gpu_v2_1(uchar* img_in, uchar* img_out, int width, int height, int W) {
    int img_size = height * width;
    int radius = W / 2;
    int window_size = W * W;

    thrust::device_vector<unsigned int> d_windows(img_size * window_size);
    thrust::device_vector<uchar> d_img_in(img_in, img_in + img_size);

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
