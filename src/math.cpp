#include "math.h"

// 각 열의 평균을 구하는 함수
std::vector<float> MeanOfMatrix(std::vector<std::vector<float>> matrix) {
	std::vector<float> result;
	for (int i = 0; i < matrix[0].size(); i++) {
		float sum = 0;
		for (int j = 0; j < matrix.size(); j++) {
			sum += matrix[j][i];
		}
		result.push_back(sum / matrix.size());
	}
	return result;
}

// 두개의 벡터의 cosine similarity를 구하는 함수
float CosineSimilarityOfMatrix(const std::vector<float>& a, const std::vector<float>& b) {
	float inner_product = std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
	float norm_a = std::inner_product(a.begin(), a.end(), a.begin(), 0.0f);
	float norm_b = std::inner_product(b.begin(), b.end(), b.begin(), 0.0f);

	// 예외 처리: 벡터 크기가 0이면 코사인 유사도를 정의할 수 없음
	if (norm_a == 0.0f || norm_b == 0.0f) {
		std::cerr << "Warning: One of the vectors has zero magnitude. Returning 0.\n";
		return 0.0f;  // 또는 다른 적절한 값 (-1.0f) 반환 가능
	}

	return inner_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// 벡터의 크기를 1로 만드는 함수
std::vector<float> Normalize(std::vector<float> a) {
    float norm_a = std::inner_product(a.begin(), a.end(), a.begin(), 0.0f);
    float norm = sqrt(norm_a);

    // 예외 처리: norm이 0이면 그대로 반환
    if (norm == 0.0f) {
        std::cerr << "Warning: Zero vector detected. Returning original vector.\n";
        return a;
    }

    for (int i = 0; i < a.size(); i++) {
        a[i] /= norm;
    }
    return a;
}

// Test function
int test_math() {
	std::vector<std::vector<float>> matrix = {
		{3, 2, 3},
		{1, 5, 6},
		{5, 8, 9}
	};
	std::vector<float> mean = MeanOfMatrix(matrix);
	std::cout << "Mean of matrix:" << std::endl;
	for (const auto& m : mean) {
		std::cout << m << " ";
	}
	std::cout << std::endl;
	std::vector<float> a = { 1, 1, 1 };
	std::vector<float> b = { 0, 0, 0 };
	a = Normalize(a);
	b = Normalize(b);
	std::cout << "Cosine similarity:" << std::endl;
	std::cout << CosineSimilarityOfMatrix(a, b) << std::endl;
	return 0;
}