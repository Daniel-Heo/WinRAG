/*******************************************************************************
    파   일   명 : sentencepiece.h
    프로그램명칭 :  Sentence Piece Tokenizer
    작   성   일 : 2025.2.22
    작   성   자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    프로그램용도 : SentencePiece를 사용하여 토크나이징을 수행
                           Trie 구조를 사용하여 최적화된 토크나이저 구현
                           메모리 풀을 사용하여 속도 향상
    참 고 사 항  : before ->  Jet makers feud over seat width with big orders at stake
                        After -> _J et _makers _fe ud _over _seat _width _with _big _orders _at _stake
    라 이 센 스  : MIT License

    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  최초 생성
    ----------------------------------------------------------------------------

*******************************************************************************/
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