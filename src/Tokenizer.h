#pragma once
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <locale>
#include <codecvt>
#include <iterator>
#include <algorithm>
#include <cctype>
#include <cwchar>
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include "json.hpp"

using json = nlohmann::json;

std::unordered_map<std::string, int> load_mapping(const std::string& filename);
std::vector<std::string> tokenize(const std::string& text, const std::unordered_map<std::string, int>& token_mapping);
std::vector<int> tokenize_ids(const std::string& text, const std::unordered_map<std::string, int>& token_mapping);
