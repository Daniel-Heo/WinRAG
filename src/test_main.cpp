#include "wordpiece.h"
#include "weights.h"
#include "math.h"

int test_mapping() {
    // tokenizer 파일 로드
    loadTokenizer("tokenizer.json");

    // 테스트할 문장 입력
   // std::string text = "Deep learning improves AI models";
    std::string text = "딥러닝은 AI 모델을 개선합니다.";
    std::vector<Token> tokens;

    // 문장 토큰화
    tokens = tokenize(text);

    // 가중치 가져오기


    return 0;
}

int main() {
    //SetConsoleOutputCP(CP_UTF8);
	//test_wordpiece(); // 1.45초 // 0.033
	test_weights();
	//test_mapping();
    //test_math();
    return 0;
}


