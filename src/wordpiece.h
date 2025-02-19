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

struct Token {
    std::string key;  // 문자열
    int id;         // 정수
};

void loadTokenizer(const std::string& filename);
std::vector<Token> tokenize(const std::string& text);