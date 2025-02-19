// test_token.cpp
#include "wordpiece.h"

int main() {
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