/*******************************************************************************
    파     일     명 : tokenizer.cpp
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
#include "tokenizer.h"

using json = nlohmann::json;
using namespace std;

/****************************************************************
* Function Name: TrieNode (Constructor)
* Description: TrieNode 기본 생성자
* Parameters: 없음
****************************************************************/
Tokenizer::TrieNode::TrieNode() : isEnd(false), id(-1) { }

/****************************************************************
* Function Name: Tokenizer::MemoryPool (Constructor)
* Description: 메모리 풀 초기화
* Parameters:
*   - initialSize: 초기 할당할 노드 개수 (size_t)
****************************************************************/
Tokenizer::MemoryPool::MemoryPool(size_t initialSize) : index(0) {
    pool.resize(initialSize); // resize로 미리 공간 확보만 함 (초기화 없음)
}

/****************************************************************
* Function Name: Tokenizer::MemoryPool::allocate
* Description: 메모리 풀에서 새 Trie 노드를 할당
* Parameters: 없음
* Return: 할당된 TrieNode 포인터
****************************************************************/
Tokenizer::TrieNode* Tokenizer::MemoryPool::allocate() {
    if (index >= pool.size()) {
        size_t newSize = pool.size() * 2;
        for (size_t i = pool.size(); i < newSize; ++i) {
            pool.resize(newSize); // 확장도 resize만으로 공간 확보
        }
    }
    TrieNode* node = &pool[index++];

    // node가 처음 사용될 때만 명시적으로 초기화
    node->isEnd = false;
    node->id = -1;
#if TRIE_SEARCH_TYPE == 1
    memset(node->children, 0, sizeof(node->children));
#endif

    return node;
}

/****************************************************************
* Function Name: Tokenizer (Constructor)
* Description: Tokenizer 기본 생성자
****************************************************************/
Tokenizer::Tokenizer() : nodePool(nullptr), root(nullptr) {
}

/****************************************************************
* Function Name: ~Tokenizer (Destructor)
* Description: 동적 할당된 메모리 해제
****************************************************************/
Tokenizer::~Tokenizer() {
    delete nodePool;
}

/****************************************************************
* Function Name: loadTokenizer
* Description: JSON 형식의 tokenizer 모델을 로드하여 Trie를 구성
* Parameters: 토크나이저 모델 파일명 (const std::string&)
* Return: 
****************************************************************/
void Tokenizer::loadTokenizer(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: tokenizer.json 파일을 열 수 없습니다.\n";
        exit(1);
    }

    json tokenizer;
    file >> tokenizer;

    // 디코더 타입 감지 (Metaspace → SentencePiece, WordPiece → WordPiece)
    if (tokenizer.contains("decoder") && tokenizer["decoder"].contains("type")) {
        decoderType = tokenizer["decoder"]["type"].get<std::string>();
    }
    else {
        std::cerr << "Error: tokenizer.json에 decoder.type이 정의되지 않음.\n";
        exit(1);
    }

    // `replacement` 또는 `prefix` 값 가져오기
    if (decoderType == "Metaspace" && tokenizer["decoder"].contains("replacement")) {
        subwordPrefix = tokenizer["decoder"]["replacement"].get<std::string>();
    }
    else if (decoderType == "WordPiece" && tokenizer["decoder"].contains("prefix")) {
        subwordPrefix = tokenizer["decoder"]["prefix"].get<std::string>();
    }
    else {
        subwordPrefix = (decoderType == "Metaspace") ? "▁" : "##";  // 기본값 설정
    }

    // UNK 토큰 확인
    if (tokenizer["model"].contains("unk_token")) {
        unkToken = tokenizer["model"]["unk_token"].get<std::string>();
		unkId = tokenizer["model"]["vocab"][unkToken].get<int>();
    }
    else {
       // WordPiece인 경우
        if (decoderType == "WordPiece") {
            unkToken = "[UNK]"; // 기본값
            unkId = 1;
		}
        else {
            unkToken = "<unk>"; // 기본값
            unkId = 0;
        }
    }

    size_t vocabSize = tokenizer["model"]["vocab"].size();
    size_t estimatedNodes = vocabSize * 3;

    delete nodePool;
    nodePool = new MemoryPool(estimatedNodes);
    root = nodePool->allocate();

    for (auto it = tokenizer["model"]["vocab"].begin(); it != tokenizer["model"]["vocab"].end(); ++it) {
        std::string token = it.key();
        int token_id = it.value().get<int>();

        TrieNode* current = root;

        for (unsigned char ch : token) {
            if (!current->children[ch]) {
                current->children[ch] = nodePool->allocate();
            }
            current = current->children[ch];
        }
        current->isEnd = true;
        current->id = token_id;
    }
	//printf("Tokenizer 로드 완료\n");
}

// 
/****************************************************************
* Function Name: searchLastMatchedToken
* Description: 가장 긴 매칭된 토큰 검색
* Parameters: 단어    (const std::string&)
* Return: 가장 긴 매칭된 토큰과 ID를 반환 (std::pair<std::string, int>)
****************************************************************/
std::pair<std::string, int> Tokenizer::searchLastMatchedToken(const std::string& word, bool isSubword) const {
    TrieNode* current = root;
    int lastMatchedId = -1;
    int lastMatchedPos = -1;

    //std::string diffWord = word;

	//if (isSubword&& decoderType == "WordPiece" )
    //    diffWord = subwordPrefix + word;

    const char* ptr = word.c_str();
    for (size_t i = 0; *ptr; ++i, ++ptr) {
        unsigned char ch = static_cast<unsigned char>(*ptr);
        if (!current->children[ch]) break;

        current = current->children[ch];

        if (current->isEnd) {
            lastMatchedId = current->id;
            lastMatchedPos = i;
        }
    }

    if (lastMatchedId != -1) return std::make_pair(word.substr(0, lastMatchedPos + 1), lastMatchedId);

    return { "", -1 };
}

/****************************************************************
* Function Name: tokenize
* Description: 입력 문장을 토큰화하여 토큰 목록을 반환 (Trie 기반 Greedy Matching)
* Parameters: 입력 문장 (const std::string&)
* Return: 토큰 목록 (std::vector<Token>)
* ***************************************************************/
std::vector<Token> Tokenizer::tokenize(const std::string& text) const {
    std::vector<Token> tokens;
    std::istringstream iss(text);
    std::vector<std::string> words(
        (std::istream_iterator<std::string>(iss)),
        std::istream_iterator<std::string>());

    for (const auto& word : words) {
        std::string input = (decoderType == "WordPiece") ? word : subwordPrefix + word;
        size_t position = 0;
        size_t end = input.size();
        bool isSubword = false;

        while (position < end) {
            std::string substring = input.substr(position, end - position);
            int subwordPrefixSize=0;
            if (isSubword && decoderType == "WordPiece") {
                substring = subwordPrefix + substring;
				subwordPrefixSize = subwordPrefix.size();
            }
			//printf("substring: %s\n", substring.c_str());
            auto matchedToken = searchLastMatchedToken(substring, isSubword);

            if (matchedToken.second != -1) {
                // WordPiece 또는 SentencePiece 방식에 따라 접두어 추가
                //std::string tokenKey = (isSubword) ? subwordPrefix + matchedToken.first : matchedToken.first;
                tokens.push_back(Token{ matchedToken.first, matchedToken.second });
                position += matchedToken.first.size()- subwordPrefixSize;
                isSubword = true; // 이후 토큰은 서브워드 처리
            }
            else {
                tokens.push_back(Token{ unkToken, unkId });

                char c = input[position];
                int byteCount;
                if ((c & 0x80) == 0) byteCount = 1;
                else if ((c & 0xE0) == 0xC0) byteCount = 2;
                else if ((c & 0xF0) == 0xE0) byteCount = 3;
                else if ((c & 0xF8) == 0xF0) byteCount = 4;
                else byteCount = 1;

                if ((byteCount + position) > end)
                    byteCount = end - position;

                position += byteCount;
                isSubword = true; // 이후 토큰은 서브워드 처리
            }
        }
    }

    return tokens;
}

// 토크나이저 테스트 함수
int test_tokenizer() {
    Tokenizer tokenizer;
    tokenizer.loadTokenizer("tokenizer.json");

    // 테스트할 문장 입력
   // std::string text = "Deep learning improves AI models";
	std::string text = "딮러닝은 AI 모델을 개선합니다.";
    std::vector<Token> tokens;

    // 문장 토큰화
    clock_t start, end;
    start = clock();
	for (int i = 0; i < 100000; i++)
        tokens = tokenizer.tokenize(text);
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

int test_wordpiece() {
    Tokenizer tokenizer;
    tokenizer.loadTokenizer("tokenizer_wordpiece.json");

    std::string text = "Deep learning advances AI models";
    auto tokens = tokenizer.tokenize(text);

    for (const auto& token : tokens) {
        std::cout << token.key << ":" << token.id << " ";
    }
    std::cout << std::endl;

    return 0;
}
