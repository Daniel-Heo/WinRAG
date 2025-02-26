/*******************************************************************************
    파     일     명 : weight_tokenizer.cpp
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
#include "weight_tokenizer.h"

/****************************************************************
* Function Name: to_string (1D vector)
* Description: 1D 벡터를 문자열로 변환
* Parameters:
*   - vec: 변환할 벡터 (std::vector<float>)
* Return: 변환된 문자열 (std::string)
****************************************************************/
std::string to_string(const std::vector<float>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i != vec.size() - 1) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

/****************************************************************
* Function Name: to_string (2D vector)
* Description: 2D 벡터를 문자열로 변환
* Parameters:
*   - vec2d: 변환할 2D 벡터 (std::vector<std::vector<float>>)
* Return: 변환된 문자열 (std::string)
****************************************************************/
std::string to_string(const std::vector<std::vector<float>>& vec2d) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec2d.size(); ++i) {
        oss << to_string(vec2d[i]);  // 1D 벡터를 문자열로 변환
        if (i != vec2d.size() - 1) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

/****************************************************************
* Function Name: WeightTokenizer (Constructor)
* Description: 가중치 데이터와 토크나이저 데이터를 로드하는 생성자
* Parameters:
*   - weight_filename: 가중치 데이터 파일 경로 (std::string)
*   - tokenizer_filename: 토크나이저 파일 경로 (std::string)
****************************************************************/
WeightTokenizer::WeightTokenizer(const std::string& weight_filename, const std::string& tokenizer_filename)
{
    try {
        // 가중치 파일 로드
        weights_ = std::make_unique<WeightLoader>(weight_filename);

        // 토크나이저 파일 로드
        tokenizer_ = std::make_unique<Tokenizer>();
        tokenizer_->loadTokenizer(tokenizer_filename);
    }
    catch (const std::exception& e) {
        std::cerr << "초기화 오류: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

/****************************************************************
* Function Name: GetWeight
* Description: 입력된 텍스트의 평균 가중치 벡터를 반환
* Parameters:
*   - text: 분석할 텍스트 (std::string)
* Return: 평균 가중치 벡터 (std::vector<float>)
****************************************************************/
std::vector<float> WeightTokenizer::GetWeight(const std::string& text)
{
    // 1. 입력 텍스트를 토큰화
    std::vector<Token> tokens = tokenizer_->tokenize(text);

    // 2. 토큰 ID에 해당하는 가중치 벡터들을 수집
    std::vector<std::vector<float>> token_weights;
    for (const auto& token : tokens) {
        // 각 토큰의 ID가 weights_의 row 인덱스라고 가정합니다.
        int token_id = token.id;

        // 유효한 토큰 ID 인덱스인지 검사
        if (token_id >= 0 && token_id < weights_->get_rows()) {
            token_weights.push_back(weights_->get(token_id));
        }
        else {
            std::cerr << "잘못된 토큰 ID: " << token_id << std::endl;
        }
    }

    if (token_weights.empty()) {
        std::cerr << "토큰에 해당하는 가중치를 찾을 수 없습니다." << std::endl;
        return {};
    }

    // 3. 수집한 벡터들의 평균을 계산하여 반환
    return MeanVector(token_weights);
}

// 테스트 함수
int test_weight_tokenizer() {
    // 클래스 초기화
    WeightTokenizer weightTokenizer("embedding_weights.npy", "tokenizer.json");

    // 텍스트 입력 및 평균 가중치 벡터 계산
    std::string text = "딥러닝을 통한 자연어 처리";
    std::vector<float> averaged_weights = weightTokenizer.GetWeight(text);

    // 결과 출력
    std::cout << "입력 텍스트의 평균 가중치 벡터: " << to_string(averaged_weights) << std::endl;

    return 0;
}