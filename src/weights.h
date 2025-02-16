// weights.h

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <regex>
#include <cmath>

#define WEIGHT_SIZE 152064
#define DIM_SIZE 2048
std::vector<float> load_npy_float16_to_float32(const std::string& filename, size_t& rows, size_t& cols);
