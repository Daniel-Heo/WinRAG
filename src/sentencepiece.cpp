/*******************************************************************************
	파   일   명 : sentencepiece.cpp
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
#include "sentencepiece.h"

using json = nlohmann::json;
using namespace std;

// Trie 노드 구조체
struct TrieNode {
    bool isEnd;
    int id;
    array<TrieNode*, 256> children;

    TrieNode() : isEnd(false), id(-1) {
        children.fill(nullptr);  // 모든 포인터를 nullptr로 초기화
    }
};

// 메모리 풀 (TrieNode 할당 최적화)
struct MemoryPool {
    vector<TrieNode> pool;
    size_t index = 0;

    MemoryPool(size_t initialSize) {
        pool.reserve(initialSize);  // 초기 크기 예약
        for (size_t i = 0; i < initialSize; ++i) {
            pool.emplace_back();  // 노드 생성
        }
    }

    TrieNode* allocate() {
        if (index >= pool.size()) {
            size_t newSize = pool.size() * 2;  // 크기를 2배로 증가
            //cout << "MemoryPool 크기 증가: " << newSize << " 개의 노드 할당" << endl;
            for (size_t i = pool.size(); i < newSize; ++i) {
                pool.emplace_back();
            }
        }
        return &pool[index++];
    }
};

// Trie 루트 노드 및 메모리 풀 (초기에는 nullptr)
static MemoryPool* nodePool = nullptr;
static TrieNode* root = nullptr;

// Tokenizer.json 로드 (메모리 풀을 동적으로 설정)
void loadTokenizer(const std::string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: tokenizer.json 파일을 열 수 없습니다." << endl;
        exit(1);
    }

    json tokenizer;
    file >> tokenizer;

    // Vocab 크기 계산
    size_t vocabSize = tokenizer["model"]["vocab"].size();
    size_t estimatedNodes = vocabSize * 3; // 예비 공간 포함 (2배 여유)

    // 동적 MemoryPool 생성
    if (nodePool) delete nodePool;  // 기존 메모리 풀 해제
    nodePool = new MemoryPool(estimatedNodes);

    // Trie 루트 노드 할당
    root = nodePool->allocate();

    // Vocab 로드 (Trie에 직접 삽입)
    for (auto it = tokenizer["model"]["vocab"].begin(); it != tokenizer["model"]["vocab"].end(); ++it) {
        string token = it.key();
        int token_id = it.value().get<int>();

        TrieNode* current = root;
        for (unsigned char ch : token) { // `char` 기반 최적화
            if (!current->children[ch]) {
                current->children[ch] = nodePool->allocate(); // 메모리 풀에서 할당
            }
            current = current->children[ch];
        }
        current->isEnd = true;
        current->id = token_id;
    }

    //cout << "Tokenizer 로드 완료\n";
    //cout << "Vocab 크기: " << vocabSize << " 개\n";
}

// 마지막으로 매칭된 토큰과 ID 반환 (최적화 버전)
pair<string, int> searchLastMatchedToken(const string& word) {
    TrieNode* current = root;
    int lastMatchedId = -1;
    int lastMatchedPos = -1;  // 마지막으로 매칭된 위치 저장

    // `word[i]` 접근 최적화 → `const char*` 포인터 사용
    const char* ptr = word.c_str();
    for (size_t i = 0; *ptr; ++i, ++ptr) {
        unsigned char ch = static_cast<unsigned char>(*ptr);

        // 배열 기반 접근 (`std::array<TrieNode*, 256>`)
        if (!current->children[ch]) break;

        current = current->children[ch];

        // 단어가 완성되었으면 저장
        if (current->isEnd) {
            lastMatchedId = current->id;
            lastMatchedPos = i;
        }
    }

    // 마지막으로 매칭된 위치에서 원본 문자열 생성 (O(1) 접근)
    return (lastMatchedId != -1) ? make_pair(word.substr(0, lastMatchedPos + 1), lastMatchedId) : make_pair("", -1);
}


// Trie 기반 Greedy Matching Tokenizer (Token 반환)
std::vector<Token> tokenize(const std::string& text) {
    std::vector<Token> tokens;
    pair<string, int> matchedToken;

    // 공백으로 단어 분리 후 '▁' 추가
    std::istringstream iss(text);
    std::vector<std::string> words((std::istream_iterator<std::string>(iss)),
        std::istream_iterator<std::string>());

    //std::cout << "\n[Greedy Matching with Trie Debug]\n";
    for (const auto& word : words) {
        std::string input = "▁" + word;  // SentencePiece 스타일
        size_t position = 0;

        // Greedy Matching: 긴 토큰부터 탐색
        std::string substring;
        int end;
        char c;
		int byteCount;

        end = input.size();
        while (position < end) {

            substring = input.substr(position, end - position);
            //std::cout << "search word: " << substring << "\n";
            //printf("position: %d, end: %d ", position, end);

            // B+TREE 매칭
            matchedToken = searchLastMatchedToken(substring);
            if (matchedToken.second != -1) {
                //std::cout << " [Match]\n";
                tokens.push_back(Token{ matchedToken.first, matchedToken.second });
                position += static_cast<int>(matchedToken.first.size());

                //std::cout << "  selected token: '" << matchedToken.first << "', " << matchedToken.second << "\n";
            }
            else {
                //std::cout << " [No Match]\n";
                tokens.push_back(Token{ "<unk>", 5 });
                //std::cout << "  selected token: '" << "<unk>" << "', " << 5 << "\n";

                // utf-8 현재 문자 사이즈 계산
				c = input[position];
				if ((c & 0x80) == 0) byteCount = 1; // ASCII character
				else if ((c & 0xE0) == 0xC0) byteCount = 2; // 2-byte character
				else if ((c & 0xF0) == 0xE0) byteCount = 3; // 3-byte character
				else if ((c & 0xF8) == 0xF0) byteCount = 4; // 4-byte character
				else byteCount = 1; // Invalid UTF-8 sequence
				if ((byteCount + position) > end) byteCount = end - position;
                position += byteCount;
            }
        }
    }

    return tokens;
}

int test_sentencepiece() {
    std::string filename = "tokenizer.json";  // Python에서 저장한 매핑 파일

    //SetConsoleOutputCP(CP_UTF8);

    // tokenizer 파일 로드
    loadTokenizer(filename);

    // 테스트할 문장 입력
   // std::string text = "Deep learning improves AI models";
	std::string text = "딮러닝은 AI 모델을 개선합니다.";
    std::vector<Token> tokens;

    // 문장 토큰화
    clock_t start, end;
    start = clock();
	for (int i = 0; i < 100000; i++)
        tokens = tokenize(text);
	end = clock();
	printf("실행시간: %f\n", (double)(end - start) / CLOCKS_PER_SEC);

    // 결과 출력
    std::cout << "Tokenized Result:" << std::endl;
	for (const auto& token : tokens) {
		std::cout << token.key << ":";
		std::cout << token.id << " ";
	}
    std::cout << std::endl;

    return 0;
}
