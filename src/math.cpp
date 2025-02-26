/*******************************************************************************
    파     일     명 : math.cpp
    프로그램명칭 :  vector 연산
    프로그램용도 : 노멀라이징, 평균 계산, 코사인 유사도 계산
    참  고  사  항  :

    작    성    자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    라 이 센 스  : MIT License
    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  최초 생성
*******************************************************************************/
#include "math.h"

/****************************************************************
* Function Name: NormalizeVector
* Description: 벡터를 SIMD를 사용하여 노멀라이즈 (L2 정규화)
* Parameters:
*   - vec: 노멀라이즈할 벡터 포인터
*   - size: 벡터 크기
* Return: 없음
****************************************************************/
void NormalizeVector(float* vec, size_t size) {
    if (!vec) return;  // nullptr 체크
    size_t i = 0;
    float norm = 0.0f;

#if SIMD_TYPE == 1  // AVX2 + FMA3 코드
    __m256 sum = _mm256_setzero_ps(); // 256비트 레지스터(8개 float)
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(vec + i);
        sum = _mm256_fmadd_ps(v, v, sum);  // FMA3: sum += v * v
    }
    float sums[8];
    _mm256_storeu_ps(sums, sum);
    norm = sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];

#elif SIMD_TYPE == 0  // SSE2 코드
    __m128 sum = _mm_setzero_ps();
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(vec + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(v, v));
    }
    float sums[4];
    _mm_storeu_ps(sums, sum);
    norm = sums[0] + sums[1] + sums[2] + sums[3];

#endif

    // 남은 요소 처리
    for (; i < size; i++) norm += vec[i] * vec[i];

    norm = sqrtf(norm); // L2 노름 계산
    if (norm > 0) {
#if SIMD_TYPE == 1  // AVX2 + FMA3 정규화
        __m256 normVec = _mm256_set1_ps(1.0f / norm);
        for (i = 0; i + 7 < size; i += 8) {
            __m256 v = _mm256_loadu_ps(vec + i);
            _mm256_storeu_ps(vec + i, _mm256_mul_ps(v, normVec));
        }
#elif SIMD_TYPE == 0  // SSE2 정규화
        __m128 normVec = _mm_set1_ps(1.0f / norm);
        for (i = 0; i + 3 < size; i += 4) {
            __m128 v = _mm_loadu_ps(vec + i);
            _mm_storeu_ps(vec + i, _mm_mul_ps(v, normVec));
        }
#endif
        for (; i < size; i++) vec[i] /= norm; // 나머지 처리
    }
}

/****************************************************************
* Function Name: MeanVector
* Description: 벡터를 SIMD를 사용하여 평균을 낸다.
* Parameters:
*   - matrix: 평균을 계산할 벡터
* Return: 없음
* Date: 2025-02-21
****************************************************************/
std::vector<float> MeanVector(std::vector<std::vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<float> result(cols, 0.0f);

    for (int i = 0; i < cols; i++) {
#if SIMD_TYPE == 1  // AVX2 + FMA3 (8 floats씩 처리, 4회 언롤링)
        __m256 sum_vec1 = _mm256_setzero_ps();
        __m256 sum_vec2 = _mm256_setzero_ps();
        __m256 sum_vec3 = _mm256_setzero_ps();
        __m256 sum_vec4 = _mm256_setzero_ps();
        int j = 0;

        // 한 번에 32개씩 처리 (8 floats × 4회 언롤링)
        for (; j + 31 < rows; j += 32) {
            sum_vec1 = _mm256_add_ps(sum_vec1, _mm256_set_ps(
                matrix[j + 7][i], matrix[j + 6][i], matrix[j + 5][i], matrix[j + 4][i],
                matrix[j + 3][i], matrix[j + 2][i], matrix[j + 1][i], matrix[j + 0][i]));

            sum_vec2 = _mm256_add_ps(sum_vec2, _mm256_set_ps(
                matrix[j + 15][i], matrix[j + 14][i], matrix[j + 13][i], matrix[j + 12][i],
                matrix[j + 11][i], matrix[j + 10][i], matrix[j + 9][i], matrix[j + 8][i]));

            sum_vec3 = _mm256_add_ps(sum_vec3, _mm256_set_ps(
                matrix[j + 23][i], matrix[j + 22][i], matrix[j + 21][i], matrix[j + 20][i],
                matrix[j + 19][i], matrix[j + 18][i], matrix[j + 17][i], matrix[j + 16][i]));

            sum_vec4 = _mm256_add_ps(sum_vec4, _mm256_set_ps(
                matrix[j + 31][i], matrix[j + 30][i], matrix[j + 29][i], matrix[j + 28][i],
                matrix[j + 27][i], matrix[j + 26][i], matrix[j + 25][i], matrix[j + 24][i]));
        }

        // 남은 8개 단위 처리
        __m256 sum_vec_remain = _mm256_setzero_ps();
        for (; j + 7 < rows; j += 8) {
            sum_vec_remain = _mm256_add_ps(sum_vec_remain, _mm256_set_ps(
                matrix[j + 7][i], matrix[j + 6][i], matrix[j + 5][i], matrix[j + 4][i],
                matrix[j + 3][i], matrix[j + 2][i], matrix[j + 1][i], matrix[j + 0][i]));
        }

        // SIMD 결과 모두 합산
        sum_vec1 = _mm256_add_ps(sum_vec1, sum_vec2);
        sum_vec3 = _mm256_add_ps(sum_vec3, sum_vec4);
        sum_vec1 = _mm256_add_ps(sum_vec1, sum_vec3);
        sum_vec1 = _mm256_add_ps(sum_vec1, sum_vec_remain);

        float temp[8];
        _mm256_storeu_ps(temp, sum_vec1);
        float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

#elif SIMD_TYPE == 0  // SSE2 사용 (4 floats씩 처리, 4회 언롤링)
        __m128 sum_vec1 = _mm_setzero_ps();
        __m128 sum_vec2 = _mm_setzero_ps();
        __m128 sum_vec3 = _mm_setzero_ps();
        __m128 sum_vec4 = _mm_setzero_ps();
        int j = 0;

        // 한 번에 16개씩 처리 (4 floats × 4회 언롤링)
        for (; j + 15 < rows; j += 16) {
            sum_vec1 = _mm_add_ps(sum_vec1, _mm_set_ps(
                matrix[j + 3][i], matrix[j + 2][i], matrix[j + 1][i], matrix[j + 0][i]));

            sum_vec2 = _mm_add_ps(sum_vec2, _mm_set_ps(
                matrix[j + 7][i], matrix[j + 6][i], matrix[j + 5][i], matrix[j + 4][i]));

            sum_vec3 = _mm_add_ps(sum_vec3, _mm_set_ps(
                matrix[j + 11][i], matrix[j + 10][i], matrix[j + 9][i], matrix[j + 8][i]));

            sum_vec4 = _mm_add_ps(sum_vec4, _mm_set_ps(
                matrix[j + 15][i], matrix[j + 14][i], matrix[j + 13][i], matrix[j + 12][i]));
        }

        // 남은 4개 단위 처리
        __m128 sum_vec_remain = _mm_setzero_ps();
        for (; j + 3 < rows; j += 4) {
            sum_vec_remain = _mm_add_ps(sum_vec_remain, _mm_set_ps(
                matrix[j + 3][i], matrix[j + 2][i], matrix[j + 1][i], matrix[j + 0][i]));
        }

        // SIMD 결과 모두 합산
        sum_vec1 = _mm_add_ps(sum_vec1, sum_vec2);
        sum_vec3 = _mm_add_ps(sum_vec3, sum_vec4);
        sum_vec1 = _mm_add_ps(sum_vec1, sum_vec3);
        sum_vec1 = _mm_add_ps(sum_vec1, sum_vec_remain);

        float temp[4];
        _mm_storeu_ps(temp, sum_vec1);
        float sum = temp[0] + temp[1] + temp[2] + temp[3];

#else  // SIMD 미사용 일반 코드 (범용)
        float sum = 0.0f;
        int j = 0;
#endif
        // 남은 요소 일반 처리 (공통)
        for (; j < rows; j++) {
            sum += matrix[j][i];
        }

        result[i] = sum / rows;
    }
    return result;
}

/****************************************************************
* Function Name: CosineSimilarity
* Description: 두 벡터 간 코사인 유사도를 SIMD로 계산 (노멀라이즈된 벡터 가정)
* Parameters:
*   - v1: 첫 번째 벡터 포인터
*   - v2: 두 번째 벡터 포인터
*   - size: 벡터 크기
* Return: 코사인 유사도 값 (float)
* Date: 2025-02-21
****************************************************************/
float CosineSimilarity(const float* v1, const float* v2, size_t size) {
    if (!v1 || !v2) return 0.0f; // nullptr 체크
    size_t i = 0;
    float dot = 0.0f;

#if SIMD_TYPE == 1  // AVX2 + FMA3 코드 (루프 언롤링 적용)
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();

    size_t simdBlockSize = 32;  // 8 floats * 4 (언롤링 4회)
    size_t simdEnd = size - (size % simdBlockSize);

    for (; i < simdEnd; i += simdBlockSize) {
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(v1 + i), _mm256_loadu_ps(v2 + i), sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(v1 + i + 8), _mm256_loadu_ps(v2 + i + 8), sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(v1 + i + 16), _mm256_loadu_ps(v2 + i + 16), sum3);
        sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(v1 + i + 24), _mm256_loadu_ps(v2 + i + 24), sum4);
    }

    // 부분합 계산
    sum1 = _mm256_add_ps(sum1, sum2);
    sum3 = _mm256_add_ps(sum3, sum4);
    sum1 = _mm256_add_ps(sum1, sum3);

    float sums[8];
    _mm256_storeu_ps(sums, sum1);

    dot = sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];

    // 남은 요소 처리
    for (; i < size; i++) dot += v1[i] * v2[i];

#elif SIMD_TYPE == 0  // SSE2 코드 (언롤링 4회 적용)
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    __m128 sum3 = _mm_setzero_ps();
    __m128 sum4 = _mm_setzero_ps();

    size_t simdBlockSize = 16;  // 4 floats * 4
    size_t simdEnd = size - (size % simdBlockSize);

    for (; i < simdEnd; i += simdBlockSize) {
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_loadu_ps(v1 + i), _mm_loadu_ps(v2 + i)));
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(_mm_loadu_ps(v1 + i + 4), _mm_loadu_ps(v2 + i + 4)));
        sum3 = _mm_add_ps(sum3, _mm_mul_ps(_mm_loadu_ps(v1 + i + 8), _mm_loadu_ps(v2 + i + 8)));
        sum4 = _mm_add_ps(sum4, _mm_mul_ps(_mm_loadu_ps(v1 + i + 12), _mm_loadu_ps(v2 + i + 12)));
    }

    // 부분합 계산
    sum1 = _mm_add_ps(sum1, sum2);
    sum3 = _mm_add_ps(sum3, sum4);
    sum1 = _mm_add_ps(sum1, sum3);

    float sums[4];
    _mm_storeu_ps(sums, sum1);
    dot = sums[0] + sums[1] + sums[2] + sums[3];

    // 남은 요소 처리
    for (; i < size; i++) dot += v1[i] * v2[i];

#else  // SIMD 미사용 일반 코드
    for (i = 0; i < size; i++) {
        dot += v1[i] * v2[i];
    }
#endif

    return dot; // 이미 정규화된 벡터를 가정
}

// Test function
int test_math() {
	std::vector<std::vector<float>> matrix = {
		{3, 2, 3},
		{1, 5, 6},
		{5, 8, 9}
	};
	std::vector<float> mean = MeanVector(matrix);
	std::cout << "Mean of matrix:" << std::endl;
	for (const auto& m : mean) {
		std::cout << m << " ";
	}
	std::cout << std::endl;
	float a[3] = {1, 1, 1};
	float  b[3] = {0, 0, 0};
	NormalizeVector(a, 3);
    NormalizeVector(b, 3);
	std::cout << "Cosine similarity:" << std::endl;
	std::cout << CosineSimilarity(a, b, 3) << std::endl;
	return 0;
}