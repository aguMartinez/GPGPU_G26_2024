#include <vector>
#include <algorithm>

#ifndef _UCHAR
#define _UCHAR
    typedef unsigned char uchar;
#endif

void printVector2(uchar* V, int n){
    for (int i = 0; i < n; i++){
        printf("%.1f ", V[i]);
    }
    printf("\n");
}

void filtro_mediana_cpu(uchar * img_in, uchar * img_out, int width, int height, int W) {
    int radius = W / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::vector<uchar> neighborhood;

            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        neighborhood.push_back(img_in[ny * width + nx]);
                    else 
                        neighborhood.push_back(0);
                }
            }
            // TODO: Revisar nth_elemet
            std::nth_element(neighborhood.begin(), neighborhood.begin() + neighborhood.size() / 2, neighborhood.end());
            float median = neighborhood[neighborhood.size() / 2];

            int threshold = 0;
            
            // [begin] debug
            // printf("Median: %f\n", fabs(img_in[y * width + x] - median));
            // [end] debug
            
            if (fabs(img_in[y * width + x] - median) > threshold) {
                img_out[y * width + x] = median;
            } else {
                img_out[y * width + x] = img_in[y * width + x];
            }
            
        }
    }
}

void filtro_mediana_cpu_old(float * img_in, float * img_out, int width, int height, int W) {
    int radius = W / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::vector<float> neighborhood;

            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        neighborhood.push_back(img_in[ny * width + nx]);
                    else 
                        neighborhood.push_back(0.0);
                }
            }
            // TODO: Revisar nth_elemet
            std::nth_element(neighborhood.begin(), neighborhood.begin() + neighborhood.size() / 2, neighborhood.end());
            float median = neighborhood[neighborhood.size() / 2];

            int threshold = 0;
            
            // [begin] debug
            // printf("Median: %f\n", fabs(img_in[y * width + x] - median));
            // [end] debug
            
            if (fabs(img_in[y * width + x] - median) > threshold) {
                img_out[y * width + x] = median;
            } else {
                img_out[y * width + x] = img_in[y * width + x];
            }
            
        }
    }
}