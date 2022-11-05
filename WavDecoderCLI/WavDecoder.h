#pragma once
#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#ifndef WavDecoder_H
#define WavDecoder_H

class WavDecoder {
public:
	static std::vector<long long int> Decode(std::string filepath);
};

#endif
