// test_token.cpp
#include "Tokenizer.h"

int main() {
    std::string filename = "mapping.json";  // Python에서 저장한 매핑 파일

    //SetConsoleOutputCP(CP_UTF8);

    // mapping.txt 파일 로드
    std::unordered_map<std::string, int> token_mapping = load_mapping(filename);

    // 테스트할 문장 입력
    std::string text = "Deep learning improves AI models";

    // 문장 토큰화
    std::vector<std::string> tokens = tokenize(text, token_mapping);

    // IDS로 변환
    std::vector<int> ids = tokenize_ids(text, token_mapping);

    // 결과 출력
    std::cout << "Tokenized Result:" << std::endl;
    for (const auto& token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

	std::cout << "IDS Result:" << std::endl;
	for (const auto& id : ids) {
		std::cout << id << " ";
	}
    std::cout << std::endl;

    return 0;
}