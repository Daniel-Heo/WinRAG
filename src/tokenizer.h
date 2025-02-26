/*******************************************************************************
    파     일     명 : tokenizer.h
    프로그램명칭 : Sentence Piece Tokenizer
    프로그램용도 : SentencePiece를 사용하여 토크나이징을 수행
                           Trie 구조를 사용하여 최적화된 토크나이저 구현
                           메모리 풀을 사용하여 속도 향상
    참  고  사  항  : Before ->  Jet makers feud over seat width with big orders at stake
                           After -> _J et _makers _fe ud _over _seat _width _with _big _orders _at _stake

    작    성    자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    라 이 센 스  : MIT License
    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  최초 생성
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

// Trie 검색 방식 선택 : 압축방식 Trie(Radix Trie)가 필요하면 적용을 고려해보자.
#define TRIE_SEARCH_TYPE 1 // 0: Low memory,low speed  1: High memory,high speed 

/****************************************************************
* Struct Name: Token
* Description: 토큰의 키(문자열)와 ID(정수)를 저장하는 구조체
****************************************************************/
struct Token {
    std::string key;   // 문자열
    int id;            // 정수
};

/****************************************************************
* Class Name: Tokenizer
* Description: SentencePiece & WordPiece 자동 선택
*              Trie 구조를 사용하여 검색 최적화 
****************************************************************/
class Tokenizer {
private:
    struct TrieNode {
        bool isEnd;
        int id;
#if TRIE_SEARCH_TYPE == 1 // 속도 향상을 위한 방법
        TrieNode* children[256];
#else
        std::unordered_map<char, TrieNode*> children;
#endif

        TrieNode();
    };

    struct MemoryPool {
        std::vector<TrieNode> pool;
        size_t index;

        explicit MemoryPool(size_t initialSize);
        TrieNode* allocate();
    };

    // 멤버 변수
    MemoryPool* nodePool;
    TrieNode* root;
    std::string decoderType; // "Metaspace" → SentencePiece, "WordPiece" → WordPiece
    std::string unkToken;    // UNK 토큰
	int unkId;               // UNK 토큰 ID
    std::string subwordPrefix; // SentencePiece의 replacement 값 또는 WordPiece의 prefix 값


    // 내부 함수
    std::pair<std::string, int> searchLastMatchedToken(const std::string& word, bool isSubword) const;

public:
    Tokenizer();                            // 생성자
    ~Tokenizer();                           // 소멸자 (메모리 관리)

    void loadTokenizer(const std::string& filename);
    std::vector<Token> tokenize(const std::string& text) const;
};

// 테스트 함수
int test_tokenizer();
int test_wordpiece();