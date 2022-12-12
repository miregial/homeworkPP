
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda.h>

#pragma region Wav Decoder
struct WAVHEADER
{
    char chunkId[4];
    unsigned long chunkSize;
    char format[4];
    char subchunk1Id[4];
    unsigned long subchunk1Size;
    unsigned short audioFormat;
    unsigned short numChannels;
    unsigned long sampleRate;
    unsigned long byteRate;
    unsigned short blockAlign;
    unsigned short bitsPerSample;
    char subchunk2Id[4];
    unsigned long subchunk2Size;
};

enum WavChunks {
    RiffHeader = 0x46464952,
    WavRiff = 0x54651475,
    Format = 0x020746d66,
    LabeledText = 0x478747C6,
    Instrumentation = 0x478747C6,
    Sample = 0x6C706D73,
    Fact = 0x47361666,
    Data = 0x61746164,
    Junk = 0x4b4e554a,
};

std::vector<long long int> ReadData(FILE* file, uint16_t bytesPerSample, uint16_t numSamples) {
	static const uint16_t BUFFER_SIZE = 4096;
	std::vector<long long int> data;
	size_t bytesRead;

	switch (bytesPerSample)
	{
	case 1:
		char int8;
		while (fread(&int8, sizeof(char), 1, file) > 0)
		{
			data.push_back((long long int)int8);
		}
		break;
	case 2:
		int int16;
		while (fread(&int16, sizeof(int), 1, file) > 0)
		{
			data.push_back((long long int)int16);
		}
		break;
	case 4:
		long int int32;
		while (fread(&int32, sizeof(long int), 1, file) > 0)
		{
			data.push_back((long long int)int32);
		}
		break;
	case 8:
		long long int int64;
		while (fread(&int64, sizeof(long long int), 1, file) > 0)
		{
			data.push_back(int64);
		}
		break;
	default:
		throw std::invalid_argument("PCM BitRate should be in (8, 16, 32, 64)!");
		break;
	}

	return data;
}

std::vector<long long int> Decode(std::string filepath) {
	std::vector<long long int> result;
	int headerSize = sizeof(WAVHEADER);
	WAVHEADER wavHeader;

	FILE* wavFile;
	fopen_s(&wavFile, filepath.c_str(), "rb+");
	if (wavFile == NULL)
	{
		std::cerr << "Err: File \"" << filepath << "\" not found!" << std::endl;
		exit(-1);
	}

	size_t headerBytesRead = fread(&wavHeader, 1, headerSize, wavFile);
	if (headerBytesRead > 0) {
		uint16_t bytesPerSample = wavHeader.bitsPerSample / 8;
		uint16_t numSamples = wavHeader.chunkSize / bytesPerSample;

		result = ReadData(wavFile, bytesPerSample, numSamples);
	}

	fclose(wavFile);
	return result;
}
#pragma endregion

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "Err: CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

__global__ void CountGtLeqKernel(long long int* data, long long int target, unsigned int* resLeq, unsigned int* resGt, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i >= size)
        return;
    printf("%d | [%u->%u]=%lld\n", threadIdx.x, i, size, data[i]);
    if (abs(data[i]) > target) resGt[threadIdx.x]++;
    else resLeq[threadIdx.x]++;
}

void CountGtLeqCUDA(long long int* data, unsigned int dataSize, unsigned int threadCount, long long int target, unsigned int* leq, unsigned int* gt)
{
    long long int* d_data = 0;
    unsigned int *d_leq = 0, *d_gt = 0;
    unsigned int *h_leq = new unsigned int[dataSize], *h_gt = new unsigned int[dataSize];
    dim3 gridSize = dim3(dataSize / threadCount + 1);
    dim3 blockSize = dim3(threadCount);

    cudaSetDevice(0);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMalloc((void**)&d_data, dataSize * sizeof(long long int)));
    checkCudaErrors(cudaMalloc((void**)&d_leq, threadCount * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&d_gt, threadCount * sizeof(unsigned int)));

    checkCudaErrors(cudaMemcpy(d_data, data, dataSize * sizeof(long long int), cudaMemcpyHostToDevice));
    
    CountGtLeqKernel<<<gridSize, blockSize>>>(d_data, target, d_leq, d_gt, dataSize);
    checkCudaErrors(cudaGetLastError());

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_leq, d_leq, threadCount * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_gt, d_gt, threadCount * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < threadCount; i++)
    {
        (*gt) += h_gt[i];
        (*leq) += h_leq[i];
    }

    cudaFree(d_data);
    cudaFree(d_leq);
    cudaFree(d_gt);
    delete[] h_gt;
    delete[] h_leq;
}

int main()
{
    std::string filename = "wavExample.wav";
    auto data = Decode(filename);
    
    long long int target = 16000;
    unsigned int threadCount = 1024;
    unsigned int leq = 0, unsigned int gt = 0;
    
    std::cout << "File: " << filename << ", Length of file: " << data.size() << ", Target: " << target << ", Thread Number: " << threadCount << std::endl;

    CountGtLeqCUDA(data.data(), data.size(), threadCount, target, &leq, &gt);
    std::cout << "Leq: " << leq << ", Gt: " << gt << std::endl;
    
    return 0;
}