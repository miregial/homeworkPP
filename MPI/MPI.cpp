#include <iostream>
#include "mpi.h"
#include <vector>
#include "WavHeader.h"

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
	if (wavFile == nullptr)
	{
		throw std::invalid_argument("file not found");
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

std::vector<std::vector<long long int>> make_chunks(std::vector<long long int> data, int chunk_count)
{
	std::vector<std::vector<long long int>> datas(chunk_count);

	for (int i = 0; i < data.size(); i++)
	{
		int chunk_index = i % chunk_count;
		datas[chunk_index].push_back(data[i]);
	}
	return datas;
}

void count_gt_leq(std::vector<long long int> data, long long int target, int* leq, int* gt)
{
	for (auto value : data)
	{
		if (abs(value) > target) (*gt)++;
		else (*leq)++;
	}
}

void mpi_point_to_point(int argc, char* argv[])
{
	int my_id, mpi_world_size, tag, result_leq, result_gt = 0;
	MPI_Status status;
	MPI_Request request;
	long long int target = 16000;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

	if (my_id == 0)
	{
		std::vector<long long> audio_data = Decode("file_sample.wav");
		std::vector<std::vector<long long>> audio_chunks = make_chunks(audio_data, mpi_world_size - 1);
		for (size_t i = 0; i < mpi_world_size - 1; i++)
		{
			int chunk_size = audio_chunks[i].size();
			MPI_Send(audio_chunks[i].data(), chunk_size, MPI_LONG_LONG, i + 1, i, MPI_COMM_WORLD);
		}

		for (size_t i = 0; i < mpi_world_size - 1; i++)
		{
			long long node_leq;
			long long node_gt;
			MPI_Recv(&node_leq, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(&node_gt, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			result_leq += node_leq;
			result_gt += node_gt;
		}

		std::cout << "Result leq: " << result_leq << " gt: " << result_gt << std::endl;
	}
	else
	{
		int bucketSize;
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_INT, &bucketSize);
		std::vector<long long> data;
		data.resize(bucketSize);
		MPI_Recv(data.data(), bucketSize, MPI_LONG_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		count_gt_leq(data, target, &result_leq, &result_gt);

		MPI_Send(&result_leq, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&result_gt, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
}

void mpi_all(int argc, char* argv[])
{
	int my_id, mpi_worl_size, message_tag = 0;
	MPI_Status status;
	MPI_Request request;
	long long int target = 16000;
	int leq = 0, gt = 0;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_worl_size);
	std::vector<long long> sendBuff;

	int bucketSize = 0;
	int* bucketSizes = NULL;
	if (my_id == 0)
	{
		sendBuff = Decode("file_sample.wav");
		bucketSize = sendBuff.size() / mpi_worl_size;

		bucketSizes = new int[mpi_worl_size];
		for (int i = 0; i < mpi_worl_size; i++)
		{
			bucketSizes[i] = bucketSize;
		}
	}

	MPI_Scatter(bucketSizes, 1, MPI_INT, &bucketSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	std::vector<long long> reciev_data;
	reciev_data.resize(bucketSize);

	MPI_Scatter(sendBuff.data(), bucketSize, MPI_LONG_LONG,
		reciev_data.data(), bucketSize, MPI_LONG_LONG,
		0, MPI_COMM_WORLD);

	count_gt_leq(reciev_data, target, &leq, &gt);

	int* node_leqs = NULL;
	int* node_gts = NULL;

	if (my_id == 0)
	{
		node_leqs = new int[mpi_worl_size];
		node_gts = new int[mpi_worl_size];
	}

	MPI_Gather(&leq, 1, MPI_INT, node_leqs, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&gt, 1, MPI_INT, node_gts, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (my_id == 0)
	{
		int result_leq = 0, result_gt = 0;
		for (int i = 0; i < mpi_worl_size; i++)
		{
			result_gt += node_gts[i];
			result_leq += node_leqs[i];
		}

		std::cout << "Result leq: " << result_leq << " gt: " << result_gt << std::endl;
	}

	MPI_Finalize();
}

int main(int argc, char* argv[])
{
	mpi_point_to_point(argc, argv);

	// mpi_all(argc, argv);
}