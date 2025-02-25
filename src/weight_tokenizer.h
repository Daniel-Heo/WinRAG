/*******************************************************************************
    파     일     명 : weight_tokenizer.h
    프로그램명칭 :  WeightTokenizer 클래스
    프로그램용도 : WeightLoader와 Tokenizer 클래스를 래핑하여 복잡한 사용을 단순화
    참  고  사  항  :

    작    성    자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    라 이 센 스  : MIT License
    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.23   Daniel Heo  최초 생성
*******************************************************************************/
#pragma once
#include "weight_loader.h"
#include "tokenizer.h"
#include "math.h"
#include <memory>

std::string to_string(const std::vector<float>& vec);
std::string to_string(const std::vector<std::vector<float>>& vec2d);

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

