/*******************************************************************************
    파   일   명 : test_main.cpp
	프로그램명칭 : 테스트 메인
    작   성   일 : 2025.2.22
    작   성   자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    프로그램용도 : 단위테스트 및 복합 테스트
    참 고 사 항  :
    라 이 센 스  : MIT License

    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  최초 생성
    ----------------------------------------------------------------------------

*******************************************************************************/
#include "sentencepiece.h"
#include "weights.h"
#include "cluster_db.h"
#include "data_loader.h"
#include "bm25.h"
#include "text_cluster_db.h"

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
	//test_mapping();
    //test_cluster_db();
	//test_mean();
    //test_bm25();
	//test_data_loader();
    test_text_cluster_db();
    
    return 0;
}


