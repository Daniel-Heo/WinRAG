// Tokenizer.cpp

#include "Tokenizer.h"

/**
 * JSON 파일을 로드하여 std::unordered_map<std::string, int>로 변환하는 함수
 */
std::unordered_map<std::string, int> load_mapping(const std::string& filename) {
    std::unordered_map<std::string, int> token_mapping;

    // 파일에서 JSON 데이터 읽기
    std::ifstream file(filename);
    json settings;
    if (file.is_open()) {
        try {
            file >> settings; // JSON 파싱
            file.close();
        }
        catch (const std::exception& e) {
            std::cerr << "JSON 파싱 실패: " << e.what() << std::endl;
            return token_mapping;
        }
    }
    else {
        std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
        return token_mapping;
    }

    // "tokens" 객체를 unordered_map에 저장
    if (settings.size()>0) {
        for (const auto& [key, value] : settings.items()) {
            if (value.is_number_integer()) {
                token_mapping[key] = value.get<int>();
            }
            else {
                std::cerr << "Warning: 잘못된 숫자 형식 -> " << key << std::endl;
            }
        }
    }
    else {
        std::cerr << "Warning: 'tokens' 키가 없거나 잘못된 형식입니다." << std::endl;
    }

    return token_mapping;
}


/**
 * @brief Greedy Matching 기반 Tokenizer
 * - `token_mapping`(`std::unordered_map`)을 사용해 가장 긴 토큰을 찾음
 * - Vocab에 없으면 한 글자씩 분리
 *
 * @param text 입력 텍스트
 * @param token_mapping 토큰 매핑 (std::unordered_map)
 * @return std::vector<std::string> 토큰 리스트
 */
std::vector<std::string> tokenize(const std::string& text, const std::unordered_map<std::string, int>& token_mapping) {
    std::vector<std::string> tokens;

    // 공백으로 단어 분리 후 각 단어에 '▁' 추가
    std::istringstream iss(text);
    std::vector<std::string> words((std::istream_iterator<std::string>(iss)),
        std::istream_iterator<std::string>());

    std::cout << "\n[Greedy Matching Debug]\n";
    for (const auto& word : words) {
        std::string input = "▁" + word;  // SentencePiece 스타일 ('▁' 붙이기)
        size_t position = 0;

        std::cout << "\nword: " << input << "\n";

        // Greedy Matching (오른쪽에서 한 글자씩 줄이며 탐색)
        while (position < input.size()) {
            std::string matched_token;
            int matched_length = 0;

            std::cout << "start pos: " << position << "\n";

            // 🔍 긴 문자열부터 오른쪽에서 줄여가며 탐색
            for (size_t end = input.size(); end > position; --end) {
                std::string substring = input.substr(position, end - position);

                std::cout << "    compare: '"
                    << substring << "'";

                if (token_mapping.find(substring) != token_mapping.end()) {
                    std::cout << " [Match]\n";
                    matched_token = substring;
                    matched_length = (int)substring.size();
                    break; // 가장 긴 매칭 발견 후 중단
                }
                else {
                    std::cout << " [No Match]\n";
                }
            }

            // 매칭 성공 시
            if (!matched_token.empty()) {
                std::cout << "  selected token: '" << matched_token
                    << "' (length: " << matched_length << ")\n";
                tokens.push_back(matched_token);
                position += matched_length;
            }
            // 매칭 실패 시 (한 글자 출력)
            else {
                std::cout << "  단일 문자 출력: '"
                    << input[position] << "'\n";
                tokens.push_back(std::string(1, input[position]));
                position += 1;
            }
        }
    }

    return tokens;
}

std::vector<int> tokenize_ids(const std::string& text, const std::unordered_map<std::string, int>& token_mapping) {
    std::vector<int> ids;

    // 공백으로 단어 분리 후 각 단어에 '▁' 추가
    std::istringstream iss(text);
    std::vector<std::string> words((std::istream_iterator<std::string>(iss)),
        std::istream_iterator<std::string>());

    std::cout << "\n[Greedy Matching Debug]\n";
    for (const auto& word : words) {
        std::string input = "▁" + word;  // SentencePiece 스타일 ('▁' 붙이기)
        size_t position = 0;

        std::cout << "\nword: " << input << "\n";

        // Greedy Matching (오른쪽에서 한 글자씩 줄이며 탐색)
        while (position < input.size()) {
            int matched_id;
            int matched_length = 0;

            std::cout << "start pos: " << position << "\n";

            // 긴 문자열부터 오른쪽에서 줄여가며 탐색
            for (size_t end = input.size(); end > position; --end) {
                std::string substring = input.substr(position, end - position);

                std::cout << "    compare: '"
                    << substring << "'";

                auto it = token_mapping.find(substring);
                if (it != token_mapping.end()) {
                    std::cout << " [Match]\n";
					matched_id = it->second;
                    matched_length = (int)substring.size();
                    break; // 가장 긴 매칭 발견 후 중단
                }
                else {
                    std::cout << " [No Match]\n";
                }
            }

            // 매칭 성공 시
            if (matched_id>0) {
                std::cout << "  selected id: '" << matched_id << ")\n";
                ids.push_back(matched_id);
                position += matched_length;
            }
            // 매칭 실패 시 (한 글자 출력)
            else {
                std::cout << "  단일 문자 출력: '"
                    << input[position] << "'\n";
                ids.push_back(0);
                position += 1;
            }
        }
    }

    return ids;
}

