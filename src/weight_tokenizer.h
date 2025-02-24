#pragma once
#include "weights.h"
#include "sentencepiece.h"
#include "math.h"
#include <memory>

int test_weight_tokenizer();

// WeightTokenizer 클래스
class WeightTokenizer {
private:
    std::unique_ptr<WeightLoader> weights_;
    std::unique_ptr<Tokenizer> tokenizer_;

public:
    // 생성자: weight 데이터 파일과 tokenizer 파일 이름으로 초기화
    WeightTokenizer(const std::string& weight_filename, const std::string& tokenizer_filename);

    // 텍스트를 입력받아 가중치의 평균을 반환
    std::vector<float> GetWeight(const std::string& text);
};

