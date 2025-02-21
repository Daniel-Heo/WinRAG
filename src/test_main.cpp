#include "sentencepiece.h"
#include "weights.h"
#include "math.h"
#include "lsh.h"

int test_mapping() {
    // tokenizer 파일 로드
    loadTokenizer("tokenizer.json");

	// 가중치 파일 로드
    WeightLoader weightsData("embedding_weights.npy");

    // 테스트할 문장 입력
   // std::string text = "Deep learning improves AI models";
    std::string text = "딥러닝은 AI 모델을 개선합니다.";
    std::vector<Token> tokens;

    // 문장 토큰화
    tokens = tokenize(text);

	// 토큰으로 가중치 가져오기 
	std::vector<std::vector<float>> weights;
	int lenCount;
	for (const auto& token : tokens) {
		std::cout << token.key << " ";
		weights.push_back(weightsData.get(token.id));
		std::cout << "Weight Data[";
		lenCount = 0;
		for (const auto& w : weightsData.get(token.id)) {
			std::cout << w << " ";
			lenCount++;
			if (lenCount > 10) break;
		}
		std::cout << "]" << std::endl;
    }

    return 0;
}

int main() {
    //SetConsoleOutputCP(CP_UTF8);
    
	//test_sentencepiece(); // 1.45초 // 0.033
	//test_weights();
	//test_math();
	//test_mapping();
	test_lsh();
    
    return 0;
}


