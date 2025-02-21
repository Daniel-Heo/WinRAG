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

// 토크나이저 파일 로드
void loadTokenizer(const std::string& filename);
// 텍스트를 입력받아 토큰화된 결과를 반환
std::vector<Token> tokenize(const std::string& text);

// 테스트 함수
int test_sentencepiece();