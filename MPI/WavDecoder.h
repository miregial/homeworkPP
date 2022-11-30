#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdint>
#ifndef WavDecoder_H
#define WavDecoder_H

std::vector<long long int> Decode(std::string filepath);

#endif