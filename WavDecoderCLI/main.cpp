// Тулеубаева Айгерим М22-524

#include <iostream>
#include "WavDecoder.h"
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

std::vector<std::vector<long long int>> make_chunks(std::vector<long long int> data, int chunk_count);
void count_gt_leq(std::vector<long long int> data, long long int target, int *leq, int *gt);
void count_gt_leq_std_thread(std::vector<long long int> data, long long int target, int *leq, int *gt, int thread_count);
void count_gt_leq_CreateThread(std::vector<long long int> data, long long int target, int *leq, int *gt, int thread_count);
void count_gt_leq_pthread_create(std::vector<long long int> data, long long int target, int *leq, int *gt, int thread_count);
void count_gt_leq_fork(std::vector<long long int> data, long long int target, int *leq, int *gt, int process_count);
void child_process(int argc, char *argv[]);
void parrent_process(int argc, char *argv[]);

std::string FIFO_NAME = "undefined";

int main(int argc, char *argv[])
{
	parrent_process(argc, argv);
	// switch (pid = fork())
	// {
	// case -1:
	//	std::cerr << "Произошла ошибка" << std::endl;
	//	exit(1);
	// case 0:
	//	child_process(argc, argv);
	//	break;
	// default:
	// Родительский процесс
	//	parrent_process(argc, argv);
	//	break;
	//}
}

void parrent_process(int argc, char *argv[])
{
	if (argc < 1)
	{
		std::cerr << "Передайте аргументом путь к файлу!" << std::endl;
		exit(1);
	}

	long long int target = 16000;
	int leq = 0, gt = 0;
	int process_count = 4;
	auto data = WavDecoder::Decode(argv[1]);

	// Обычная реализация
	// count_gt_leq(data, target, &leq, &gt);

	// Реализация через std::threads
	// count_gt_leq_std_thread(data, target, &leq, &gt, 4);

	// Реализация через winApi CreateThread
	// count_gt_leq_CreateThread(data, target, &leq, &gt, 4);

	// Реализация через Linux pthread_create
	// count_gt_leq_pthread_create(data, target, &leq, &gt, 4);

	// Реализация через fork для linux
	count_gt_leq_fork(data, target, &leq, &gt, 4);

	std::cout << "Length: " << data.size() << std::endl;
	std::cout << "Target: " << target << std::endl;
	std::cout << "Less or equal than target: " << leq << std::endl;
	std::cout << "Great than target: " << gt << std::endl;
	std::cout << "Sum of leq and gt: " << gt + leq << std::endl;
}

void child_process(int argc, char *argv[])
{
	for (int i = 0; i < argc; i++)
	{
		std::cout << "CHILD " << argv[i] << std::endl;
	}

	std::cout << FIFO_NAME << std::endl;
}

struct params
{
	std::vector<long long int> data;
	long long int target;
	int leq;
	int gt;
};

struct params_pthread
{
	std::vector<long long int> data;
	long long int target;
	int *leq;
	int *gt;
};

/// <summary>
/// Реализация задания без парралельности
/// </summary>
/// <param name="data">Массив данных</param>
/// <param name="target">Значения относительно которого мы проверяем</param>
/// <param name="leq">Значения по ссылке, возвращаемый параметр </param>
/// <param name="gt"></param>
void count_gt_leq(std::vector<long long int> data, long long int target, int *leq, int *gt)
{
	for (auto value : data)
	{
		if (abs(value) > target)
			(*gt)++;
		else
			(*leq)++;
	}
}

#ifdef __linux__
void *count_gt_leq_thread_safe(void *args)
{
	params_pthread *param = (params_pthread *)args;
	for (auto value : param->data)
	{
		// Использовавания мютексов по заданию, для большей произоводительности их не нужно использовать, а передавать разные ссылки
		pthread_mutex_lock(&mutex);
		if (abs(value) > param->target)
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
void count_gt_leq_std_thread(std::vector<long long int> data, long long int target, int *leq, int *gt, int thread_count)
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
DWORD __stdcall count_gt_leq_wrapper(void *args)
{
	params *param = (params *)args;
	count_gt_leq(param->data, param->target, &(param->leq), &(param->gt));
	return 0;
}
#endif

void count_gt_leq_CreateThread(std::vector<long long int> data, long long int target, int *leq, int *gt, int thread_count)
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

		auto handle = CreateThread(NULL, 0, &count_gt_leq_wrapper, (void *)&params[i], 0, NULL);

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

void count_gt_leq_pthread_create(std::vector<long long int> data, long long int target, int *leq, int *gt, int thread_count)
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

		int res = pthread_create(&threads_id[i], NULL, &count_gt_leq_thread_safe, (void *)&params[i]);

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
struct fork_result {
   int leq;
   int gt;
};

void count_gt_leq_fork(std::vector<long long int> data, long long int target, int *leq, int *gt, int process_count)
{
#ifdef __linux__
	auto chunks = make_chunks(data, process_count);

	std::vector<pid_t> pids(process_count);
	std::vector<std::string> files(process_count); // дискрипторы именнованного канала

	for (int i = 0; i < process_count; i++)
	{
        FIFO_NAME = "fifo" + std::to_string(i);

		// mknod(FIFO_NAME.c_str(), S_IFIFO | 0666, 0);

        files[i] = FIFO_NAME;
		// fds[i] = open(FIFO_NAME.c_str(), O_WRONLY);

        auto chunk = chunks[i];
		pids[i] = fork();
        if (pids[i] == -1) 
        {
            std::cerr << "Can't create child process" << std::endl;
            exit(-1);
        }

		if (pids[i] == 0)
		{
			// Дочерний процесс
            fork_result res;
            res.leq = 0;
            res.gt = 0;
            count_gt_leq(chunk, target, &res.leq, &res.gt);

            std::cout << "Child process with pid: " << getpid() << " get chunk length: " << chunk.size();
            std::cout << "\tLEQ: " << res.leq << "\tGT:" << res.gt << "\tTARGET:" << target << std::endl;
            FILE *f = fopen(files[i].c_str(), "wb");
            fwrite(&res, sizeof(fork_result), 1, f);
            fclose(f);
            sleep(2);
			exit(0);
		}
	}
    
    for (int i = 0; i < process_count; i++) 
    {
        int status;
        std::cout << "Ждем процесс " << pids[i] << std::endl;
        if(waitpid(pids[i], &status, 0) == -1) {
            std::cerr << "Ошибка ожидание процесса " << pids[i] << std::endl; 
        } else {
            if (WIFEXITED(status)) {
                // std::cout << "Потомок закончил работу с кодом " << WEXITSTATUS(status) << std::endl;
            } else if (WIFSIGNALED(status)) {
                // std::cout << "Потомок был убит сигналом " << WTERMSIG(status) << std::endl;
            }
        }

        FILE *f = fopen(files[i].c_str(), "rb");

        fork_result res;
        size_t readed_byte = fread(&res, 1, sizeof(fork_result), f);
        // std::cout << "Read from " << pids[i] << " " << readed_byte << " bytes;" << std::endl;
        *leq += res.leq;
        *gt += res.gt;
        fclose(f);
    }

#else
	std::cerr << "OS not supported" << std::endl;
#endif
}
