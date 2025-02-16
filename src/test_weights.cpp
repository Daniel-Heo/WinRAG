// test_weights.cpp
#include "weights.h"

int main() {
    try {
        size_t rows = WEIGHT_SIZE, cols = DIM_SIZE;
        static std::vector<float> embeddings = load_npy_float16_to_float32("embedding_weights.npy", rows, cols);
		// 주소 계산 ( x, y ) = x*cols+y = x*192+y

        std::cout << "첫 번째 행 일부 출력 (float16 값): ";
        for (size_t i = 0; i < 10; ++i) {
            std::cout << embeddings[i] << " ";
        }
        std::cout << "\n총 로드된 데이터 수: " << embeddings.size() << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
    }

    return 0;
}