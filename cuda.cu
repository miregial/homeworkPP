#include <iostream>
#include "bitmap.hpp"
#include <algorithm>

struct BGR {
    uint8_t b { 0 };
    uint8_t g { 0 };
    uint8_t r { 0 };
};

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "Err: CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

#define min(a, b) (a > b)? b: a
#define max(a, b) (a > b)? a: b 

__global__ 
void kernel(BGR* data, int size, int *res, BGR* grayscale)
{
    // считает кол-во красных пикселей
	int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= size) return;

	BGR pixel = data[index];
    BGR grayscalePixel;

    // приводим картинку к полутону 
    uint8_t grayPixel = (uint8_t)min(max(0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b, 0.0), 255.0);
    grayscalePixel.r = grayPixel;
    grayscalePixel.g = grayPixel;
    grayscalePixel.b = grayPixel;
    
    if (pixel.r > 128) res[index]++;
    grayscale[index] = grayscalePixel;
}

int main()
{
    BMPMini bmp;
    bmp.read("Lena.bmp");
    auto img = bmp.get();
    printf("Readed an image\n");
    cudaError_t cudaStatus;
    BGR* pixels = reinterpret_cast<BGR*>(img.data);
        
    BGR* d_data = 0;
    BGR* d_greyscale = 0;
    BGR* h_greyscale = 0;
    int* d_res = 0;
    int* h_res = 0;
    int res = 0;
    int size = img.height * img.width;
    printf("size of image: h=%d, w=%d, channels=%d", img.height, img.width, img.channels);
    const int block_size = 1024;
    const int grids = size / block_size + 1;

    h_res = new int[block_size];
    h_greyscale = new BGR[size];

    checkCudaErrors(cudaMalloc((void**)&d_data, size * sizeof(BGR)));

    checkCudaErrors(cudaMalloc((void**)&d_res, block_size * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_greyscale, size * sizeof(BGR)));

    checkCudaErrors(cudaMemcpy(d_data, pixels, size * sizeof(BGR), cudaMemcpyHostToDevice));

    printf("Allocated 4 step\n");
    kernel<<<grids, block_size>>>(d_data, size, d_res, d_greyscale);

    printf("Allocated 5 step\n");
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    printf("Allocated 6 step");

    checkCudaErrors(cudaMemcpy(h_res, d_res, block_size * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_greyscale, d_greyscale, size * sizeof(BGR), cudaMemcpyDeviceToHost));
    printf("Allocated 7 step");
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    for (size_t i = 0; i < block_size; i++)
    {
        res += h_res[i];
    }
    printf("Allocated 8 step");
    printf("Count of reds that > 128 = %d", res);
    
    int j = 0;
    for(int i = 0; i < size; i++) {
        img.data[j++] = h_greyscale[i].b;
        img.data[j++] = h_greyscale[i].g;
        img.data[j++] = h_greyscale[i].r;
    }
    bmp.write(img, "lena_gray.bmp");
}