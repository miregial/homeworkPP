#include "WavDecoder.h"
#include "WavHeader.h"
#include <stdexcept>
#include "mpi.h"

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

void mpi_p2p(int argc, char* argv[])
{
	std::cout << "MPI P2P" << std::endl;
	int myRank, mpiSize, tag = 0;
	double cpu_time_start, cpu_time_fini;
	MPI_Status status;
	MPI_Request request;
	long long int target = 16000;
	int leq = 0, gt = 0;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
	std::cout << "Process #" << myRank << "; Size: " << mpiSize << "\n";

	if (myRank == 0)
	{
		// Родитель
		// Отправляем 
		auto data = Decode("file.wav");
		auto datas = make_chunks(data, mpiSize - 1);
		std::cout << ">>> Process #" << myRank << " opened file with length " << data.size() << std::endl;
		for (size_t i = 0; i < mpiSize - 1; i++)
		{
			int bucketSize = datas[i].size();
			//MPI_Send(&bucketSize, 1, MPI_INT, i, i, MPI_COMM_WORLD);
			MPI_Isend(&datas[i], bucketSize, MPI_LONG_LONG, i + 1, i, MPI_COMM_WORLD, &request);
			//MPI_Send(&datas[i], bucketSize, MPI_LONG_LONG, i + 1, i, MPI_COMM_WORLD);
			std::cout << ">>> Process #" << myRank << " send array with length " << bucketSize << std::endl;
		}

		for (size_t i = 0; i < mpiSize - 1; i++)
		{
			long long buff_leq;
			long long buff_gt;
			MPI_Recv(&buff_leq, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(&buff_gt, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			leq += buff_leq;
			gt += buff_gt;
		}

		std::cout << ">>> Process #" << myRank << " GET RESULT \t\tLength: " << data.size();
		std::cout << "| Target: " << target;
		std::cout << "| Less or equal than target: " << leq;
		std::cout << "| Great than target: " << gt;
		std::cout << "| Sum of leq and gt: " << gt + leq << std::endl;
	}
	else
	{
		int bucketSize;
		//MPI_Recv(&bucketSize, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_INT, &bucketSize);
		std::cout << "<<< Process #" << myRank << " will be recv data with length " << bucketSize << std::endl;

		long long* buff = new long long[bucketSize];

		MPI_Recv(buff, bucketSize, MPI_LONG_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		std::cout << "<<< Process #" << myRank << " receved data with length " << bucketSize << std::endl;
		std::vector<long long> data(buff, buff + bucketSize);

		count_gt_leq(data, target, &leq, &gt);

		MPI_Isend(&leq, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD, &request);
		MPI_Isend(&gt, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD, &request);

		std::cout << ">>> Process #" << myRank << " SEND \t\tLength: " << data.size();
		std::cout << "| Target: " << target;
		std::cout << "| Less or equal than target: " << leq;
		std::cout << "| Great than target: " << gt;
		std::cout << "| Sum of leq and gt: " << gt + leq << std::endl;
		delete[] buff;
	}

	std::cout << ">>> Process #" << myRank << " END!" << std::endl;

	MPI_Finalize();
}


void mpi_mult(int argc, char* argv[]) 
{
	std::cout << "MPI P2P" << std::endl;
	int myRank, mpiSize, tag = 0;
	double cpu_time_start, cpu_time_fini;
	MPI_Status status;
	MPI_Request request;
	long long int target = 16000;
	int leq = 0, gt = 0;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
	std::cout << "Process #" << myRank << "; Size: " << mpiSize << "\n";
	std::vector<long long> sendBuff;
	
	long long* rbuff_array;
	int bucketSize = 0;
	int* bucketSizes = NULL;
	if (myRank == 0)
	{
		// Родитель
		// Отправляем размер
		sendBuff = Decode("file.wav");
		std::cout << ">>> Process #" << myRank << " opened file with length " << sendBuff.size() << std::endl;
		bucketSize = sendBuff.size() / mpiSize;

		bucketSizes = new int[mpiSize];
		for (int i = 0; i < mpiSize; i++)
		{
			bucketSizes[i] = bucketSize;
		}
	}
	
	// Получаем размер буффера который необходимо инициализировать
	MPI_Scatter(bucketSizes, 1, MPI_INT, &bucketSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	std::cout << "<<< Process #" << myRank << " reciev length for data " << bucketSize << std::endl;

	rbuff_array = new long long[bucketSize];
	
	MPI_Scatter(sendBuff.data(), bucketSize, MPI_LONG_LONG,
				rbuff_array, bucketSize, MPI_LONG_LONG,
				0, MPI_COMM_WORLD);

	std::vector<long long> rbuff(rbuff_array, rbuff_array + bucketSize);

	std::cout << "<<< Process #" << myRank << " reciev array with length " << rbuff.size() << std::endl;

	count_gt_leq(rbuff, target, &leq, &gt);

	int* sub_leqs = NULL;
	int* sub_gts = NULL;

	if (myRank == 0)
	{
		sub_leqs = new int[mpiSize];
		sub_gts = new int[mpiSize];
	}

	MPI_Gather(&leq, 1, MPI_INT, sub_leqs, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&gt, 1, MPI_INT, sub_gts, 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::cout << ">>> Process #" << myRank << " END!" << std::endl;

	if (myRank == 0)
	{
		int result_leq = 0, result_gt = 0;
		for (int i = 0; i < mpiSize; i++) 
		{
			result_gt += sub_gts[i];
			result_leq += sub_leqs[i];
		}

		std::cout << ">>> Process #" << myRank << " GET RESULT \t\tLength: " << sendBuff.size();
		std::cout << "| Target: " << target;
		std::cout << "| Less or equal than target: " << result_leq;
		std::cout << "| Great than target: " << result_gt;
		std::cout << "| Sum of leq and gt: " << result_gt + result_leq << std::endl;
	}

	delete[] rbuff_array;

	MPI_Finalize();
}