// Тулеубаева Айгерим М22-524
// БДЗ 2

#include <iostream>
#include "bitmap.hpp"
#include <vector>
#include <thread>
#include <cstdio>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>

#ifdef __linux__

#include <pthread.h>
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
#define fopen_s(pFile, filename, mode) ((*(pFile)) = fopen((filename), (mode))) == NULL

#include <unistd.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <fcntl.h>
#include <sys/stat.h>

#endif

#ifdef _WIN32
#include <Windows.h>
#endif

struct BGR {
	uint8_t b{ 0 };
	uint8_t g{ 0 };
	uint8_t r{ 0 };
};

std::vector<std::vector<BGR>> make_chunks(std::vector<BGR> data, int chunk_count);
void count_gt_leq(std::vector<BGR> data, int target, int* leq, int* gt);
void count_gt_leq_std_thread(std::vector<BGR> data, int target, int* leq, int* gt, int thread_count);
void count_gt_leq_CreateThread(std::vector<BGR> data, int target, int* leq, int* gt, int thread_count);
void count_gt_leq_pthread_create(std::vector<BGR> data, int target, int* leq, int* gt, int thread_count);

std::string FIFO_NAME = "undefined";

int main(int argc, char* argv[])
{
	if (argc <= 1)
	{
		std::cerr << "Передайте аргументом путь к файлу!" << std::endl;
		exit(1);
	}

	int target = 128;
	int leq = 0, gt = 0;
	BMPMini image;
	image.read(argv[1]);
	ImageView imageView = image.get();
	int size = imageView.width * imageView.height;
	BGR* pixels = reinterpret_cast<BGR*>(imageView.data);
	std::vector<BGR> data;
	for (size_t i = 0; i < size; i++)
	{
		data.push_back(pixels[i]);
	}
	// Обычная реализация
	// count_gt_leq(data, target, &leq, &gt);

	// Реализация через std::threads
	count_gt_leq_std_thread(data, target, &leq, &gt, 4);

	// Реализация через winApi CreateThread
	// count_gt_leq_CreateThread(data, target, &leq, &gt, 4);

	// Реализация через Linux pthread_create
	// count_gt_leq_pthread_create(data, target, &leq, &gt, 4);

	std::cout << "Length: " << data.size() << std::endl;
	std::cout << "Target: " << target << std::endl;
	std::cout << "Less or equal than target: " << leq << std::endl;
	std::cout << "Great than target: " << gt << std::endl;
	std::cout << "Sum of leq and gt: " << gt + leq << std::endl;
}

struct params
{
	std::vector<BGR> data;
	int target;
	int leq;
	int gt;
};

struct params_pthread
{
	std::vector<BGR> data;
	int target;
	int* leq;
	int* gt;
};

/// <summary>
/// Реализация задания без парралельности
/// </summary>
/// <param name="data">Массив данных</param>
/// <param name="target">Значения относительно которого мы проверяем</param>
/// <param name="leq">Значения по ссылке, возвращаемый параметр </param>
/// <param name="gt"></param>
void count_gt_leq(std::vector<BGR> data, int target, int* leq, int* gt)
{
	for (auto value : data)
	{
		if (value.r > target)
			(*gt)++;
		else
			(*leq)++;
	}
}

#ifdef __linux__
void* count_gt_leq_thread_safe(void* args)
{
	params_pthread* param = (params_pthread*)args;
	for (auto value : param->data)
	{
		// Использовавания мютексов по заданию, для большей произоводительности их не нужно использовать, а передавать разные ссылки
		pthread_mutex_lock(&mutex);
		if (value.r > param->target)
			(*(param->gt))++;
		else
			(*(param->leq))++;
		pthread_mutex_unlock(&mutex);
	}
}
#endif

/// <summary>
/// Многопоточная реализация с использоваванием std::thread
/// </summary>
/// <param name="thread_count">Кол-во потоков</param>
void count_gt_leq_std_thread(std::vector<BGR> data, int target, int* leq, int* gt, int thread_count)
{
	std::vector<std::thread> thread_list;
	auto chunks = make_chunks(data, thread_count);
	std::vector<std::pair<int, int>> params(thread_count);
	for (int i = 0; i < thread_count; i++)
	{
		thread_list.push_back(std::thread(count_gt_leq, chunks[i], target, &(params[i].first), &(params[i].second)));
	}

	for (int i = 0; i < thread_list.size(); i++)
	{
		thread_list[i].join();
	}

	for (auto param : params)
	{
		(*leq) += param.first;
		(*gt) += param.second;
	}
}

#ifdef _WIN32
DWORD __stdcall count_gt_leq_wrapper(void* args)
{
	params* param = (params*)args;
	count_gt_leq(param->data, param->target, &(param->leq), &(param->gt));
	return 0;
}
#endif

void count_gt_leq_CreateThread(std::vector<BGR> data, int target, int* leq, int* gt, int thread_count)
{
	auto chunks = make_chunks(data, thread_count);
	std::vector<params> params(thread_count);
#ifdef _WIN32
	std::vector<HANDLE> threads_handles;

	for (int i = 0; i < thread_count; i++)
	{
		params[i].data = chunks[i];
		params[i].leq = 0;
		params[i].gt = 0;
		params[i].target = target;

		auto handle = CreateThread(NULL, 0, &count_gt_leq_wrapper, (void*)&params[i], 0, NULL);

		if (handle == NULL)
		{
			throw std::exception("Не удалось создать поток");
		}

		threads_handles.push_back(handle);
	}

	WaitForMultipleObjects(threads_handles.size(), &threads_handles[0], true, INFINITE);
#else
	std::cerr << "OS not supported" << std::endl;
#endif
	for (int i = 0; i < params.size(); i++)
	{
		(*leq) += params[i].leq;
		(*gt) += params[i].gt;
	}
}

void count_gt_leq_pthread_create(std::vector<BGR> data, int target, int* leq, int* gt, int thread_count)
{
#ifdef __linux__
	auto chunks = make_chunks(data, thread_count);
	std::vector<pthread_t> threads_id(thread_count);
	std::vector<params_pthread> params(thread_count);

	for (int i = 0; i < thread_count; i++)
	{
		params[i].data = chunks[i];
		params[i].leq = leq;
		params[i].gt = gt;
		params[i].target = target;

		int res = pthread_create(&threads_id[i], NULL, &count_gt_leq_thread_safe, (void*)&params[i]);

		if (res != 0)
		{
			throw std::exception();
		}
	}

	for (int i = 0; i < thread_count; i++)
	{
		pthread_join(threads_id[i], NULL);
	}
#else
	std::cerr << "OS not supported" << std::endl;
#endif
}

std::vector<std::vector<BGR>> make_chunks(std::vector<BGR> data, int chunk_count)
{
	std::vector<std::vector<BGR>> datas(chunk_count);

	for (int i = 0; i < data.size(); i++)
	{
		int chunk_index = i % chunk_count;
		datas[chunk_index].push_back(data[i]);
	}
	return datas;
}