#include "WavDecoder.h"
#include "WavHeader.h"
#include <stdexcept>

#ifdef __linux__
#define fopen_s(pFile, filename, mode) ((*(pFile)) = fopen((filename), (mode))) == NULL
#endif

std::vector<long long int> ReadData(FILE *file, uint16_t bitsPerSample)
{
	static const uint16_t BUFFER_SIZE = 4096;
	std::vector<long long int> data;
	size_t bytesRead;

	switch (bitsPerSample)
	{
	case 8:
		char int8;
		while (fread(&int8, sizeof(char), 1, file) > 0)
		{
			data.push_back((long long int)int8);
		}
		break;
	case 16:
		int int16;
		while (fread(&int16, sizeof(int), 1, file) > 0)
		{
			data.push_back((long long int)int16);
		}
		break;
	case 32:
		long int int32;
		while (fread(&int32, sizeof(long int), 1, file) > 0)
		{
			data.push_back((long long int)int32);
		}
		break;
	case 64:
		long long int int64;
		while (fread(&int64, sizeof(long long int), 1, file) > 0)
		{
			data.push_back(int64);
		}
		break;
	default:
		throw std::invalid_argument("PCM BitRate [your:" + std::to_string(bitsPerSample) + "] should be in (8, 16, 32, 64)!");
		break;
	}

	return data;
}

std::vector<long long int> WavDecoder::Decode(std::string filepath)
{
	std::vector<long long int> result;
	int headerSize = sizeof(WAVHEADER);
	WAVHEADER wavHeader;

	FILE *wavFile;
	fopen_s(&wavFile, filepath.c_str(), "rb+");
	if (wavFile == nullptr)
	{
		throw std::invalid_argument("file not found");
	}

	size_t headerBytesRead = fread(&wavHeader, 1, headerSize, wavFile);
	if (headerBytesRead > 0)
	{
		result = ReadData(wavFile, wavHeader.bitsPerSample);
	}

	fclose(wavFile);
	return result;
}
