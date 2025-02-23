/*******************************************************************************
    파   일   명 : spherical_grid.cpp
    프로그램명칭 :  유사도 DB
    작   성   일 : 2025.2.22
    작   성   자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    프로그램용도 : spherical grid를 사용하여 코사인 유사도에 적합한 인덱싱을 구현하여 효율성을 높인다.
                           
    참 고 사 항  :
    라 이 센 스  : MIT License

    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  최초 생성
    ----------------------------------------------------------------------------

*******************************************************************************/
#include "spherical_grid.h"

/****************************************************************
* Function Name: NormalizeVector
* Description: 벡터를 SIMD를 사용하여 노멀라이즈 (L2 정규화)
* Parameters:
*   - vec: 노멀라이즈할 벡터 포인터
*   - size: 벡터 크기
* Return: 없음
* Date: 2025-02-21
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
* TODO : 32바이트 정렬된 2D 벡터 사용을 하면 성능이 향상될 수 있음 ( 테스트 해본 바로는 에러가 심함 )
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
#if SIMD_TYPE == 1  // AVX2 + FMA3 사용
        __m256 sum_vec = _mm256_setzero_ps();  // 256비트 (8개 float) 0 초기화
        int j = 0;

        // SIMD 8개씩 더하기 (개별 값 로드)
        for (; j + 8 <= rows; j += 8) {
            __m256 row_data = _mm256_set_ps(
                matrix[j + 7][i], matrix[j + 6][i], matrix[j + 5][i], matrix[j + 4][i],
                matrix[j + 3][i], matrix[j + 2][i], matrix[j + 1][i], matrix[j][i]
            );
            sum_vec = _mm256_add_ps(sum_vec, row_data);
        }

        // SIMD 결과 합산
        float temp[8];
        _mm256_storeu_ps(temp, sum_vec);  // Unaligned Store
        float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

#elif SIMD_TYPE == 0  // SSE2 사용
        __m128 sum_vec = _mm_setzero_ps();  // 128비트 (4개 float) 0 초기화
        int j = 0;

        // SIMD 4개씩 더하기 (개별 값 로드)
        for (; j + 4 <= rows; j += 4) {
            __m128 row_data = _mm_set_ps(
                matrix[j + 3][i], matrix[j + 2][i], matrix[j + 1][i], matrix[j][i]
            );
            sum_vec = _mm_add_ps(sum_vec, row_data);
        }

        // SIMD 결과 합산
        float temp[4];
        _mm_storeu_ps(temp, sum_vec);  // Unaligned Store
        float sum = temp[0] + temp[1] + temp[2] + temp[3];
#endif

        // 남은 요소 처리
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

#if SIMD_TYPE == 1  // AVX2 + FMA3 코드
    __m256 sum = _mm256_setzero_ps(); // 256비트 레지스터(8개 float)
    for (; i + 7 < size; i += 8) {
        __m256 a = _mm256_loadu_ps(v1 + i);
        __m256 b = _mm256_loadu_ps(v2 + i);
        sum = _mm256_fmadd_ps(a, b, sum);  // FMA3: sum += a * b
    }
    float sums[8];
    _mm256_storeu_ps(sums, sum);
    dot = sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];

#elif SIMD_TYPE == 0  // SSE2 코드
    __m128 sum = _mm_setzero_ps();
    for (; i + 3 < size; i += 4) {
        __m128 a = _mm_loadu_ps(v1 + i);
        __m128 b = _mm_loadu_ps(v2 + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
    }
    float sums[4];
    _mm_storeu_ps(sums, sum);
    dot = sums[0] + sums[1] + sums[2] + sums[3];

#endif

    // 남은 요소 처리
    for (; i < size; i++) dot += v1[i] * v2[i];

    return dot; // 노멀라이즈된 벡터이므로 분모 생략
}

SphericalGrid::SphericalGrid(int dimension)
    : vectorDim(dimension), numSectors(SECTOR_COUNT_PER_PAIR) {
}

int SphericalGrid::GetSectorIndex(const float* vec) const {
    int PAIR_COUNT = vectorDim / 2;  // 벡터 크기를 2로 나눈 값 (나머지는 버림)

    int hash = 0;
    float angle, sectorSize;
    int sectorIndex;
    for (int i = 0; i < PAIR_COUNT; i++) {
        angle = atan2f(vec[2 * i + 1], vec[2 * i]);  // (y, x) 기반 각도 계산
        sectorSize = 2 * 3.14159265f / SECTOR_COUNT_PER_PAIR;
        sectorIndex = static_cast<int>((angle + 3.14159265f) / sectorSize) % SECTOR_COUNT_PER_PAIR;
        hash = hash * SECTOR_COUNT_PER_PAIR + sectorIndex;  // 해싱하여 하나의 값으로 변환
    }

    return hash % numSectors;  // 전체 섹터 개수에 맞춰 인덱스 조정
}

bool SphericalGrid::Add(const std::vector<float>& vec, const char* filePath) {
    if (vec.size() != vectorDim) return false;

    WeightEntry entry(vectorDim);
    memcpy(entry.vector, vec.data(), vectorDim * sizeof(float));
    NormalizeVector(entry.vector, vectorDim);

    entry.id = nextID++;  // ID를 0부터 시작하여 +1씩 증가 (삭제된 ID 재사용 X)
    strncpy_s(entry.filePath, filePath, MAX_FILE_PATH);

    int sector = GetSectorIndex(entry.vector);
    sectorBuckets[sector].emplace_back(std::move(entry));
    return true;
}

std::vector<std::pair<const WeightEntry*, float>> SphericalGrid::FindNearest(
    const std::vector<float>& queryVec, int k)
{
    if (queryVec.size() != vectorDim) return {};

    float* normalizedQuery = static_cast<float*>(_aligned_malloc(vectorDim * sizeof(float), 16));
    if (!normalizedQuery) return {};
    memcpy(normalizedQuery, queryVec.data(), vectorDim * sizeof(float));
    NormalizeVector(normalizedQuery, vectorDim);

    int sector = GetSectorIndex(normalizedQuery);
    std::vector<std::pair<float, const WeightEntry*>> similarities;

    auto searchSector = [&](int sectorIndex) {
        if (sectorBuckets.count(sectorIndex) > 0) {
            for (const auto& entry : sectorBuckets[sectorIndex]) {
                float sim = CosineSimilarity(normalizedQuery, entry.vector, vectorDim);
                similarities.emplace_back(sim, &entry);
            }
        }
        };

    // 현재 섹터 + 인접한 섹터(+1, -1)도 검색
    searchSector(sector);
    searchSector((sector + 1) % numSectors);
    searchSector((sector - 1 + numSectors) % numSectors);

    _aligned_free(normalizedQuery);

    if (similarities.empty()) return {};

    size_t actualK = std::min(static_cast<size_t>(k), similarities.size());
    std::partial_sort(
        similarities.begin(),
        similarities.begin() + actualK,
        similarities.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<std::pair<const WeightEntry*, float>> results;
    results.reserve(actualK);
    for (size_t i = 0; i < actualK; i++) {
        results.emplace_back(similarities[i].second, similarities[i].first);
    }

    return results;
}

std::vector<std::pair<const WeightEntry*, float>> SphericalGrid::FindNearestFull(
    const std::vector<float>& queryVec, int k)
{
    if (queryVec.size() != vectorDim) return {};

    float* normalizedQuery = static_cast<float*>(_aligned_malloc(vectorDim * sizeof(float), 16));
    if (!normalizedQuery) return {};
    memcpy(normalizedQuery, queryVec.data(), vectorDim * sizeof(float));
    NormalizeVector(normalizedQuery, vectorDim);

    std::vector<std::pair<float, const WeightEntry*>> similarities;

    // **모든 벡터를 탐색 (풀스캔)**
    for (const auto& [sectorIndex, entries] : sectorBuckets) {
        for (const auto& entry : entries) {
            float sim = CosineSimilarity(normalizedQuery, entry.vector, vectorDim);
            similarities.emplace_back(sim, &entry); // 포인터 저장
        }
    }

    _aligned_free(normalizedQuery);

    if (similarities.empty()) return {};

    size_t actualK = std::min(static_cast<size_t>(k), similarities.size());
    std::partial_sort(
        similarities.begin(),
        similarities.begin() + actualK,
        similarities.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<std::pair<const WeightEntry*, float>> results;
    results.reserve(actualK);
    for (size_t i = 0; i < actualK; i++) {
        results.emplace_back(similarities[i].second, similarities[i].first); // 포인터로 저장
    }

    return results;
}

bool SphericalGrid::Delete(int id) {
    for (auto& [sector, entries] : sectorBuckets) {
        auto it = std::remove_if(entries.begin(), entries.end(),
            [id](const WeightEntry& entry) { return entry.id == id; });

        if (it != entries.end()) {
            entries.erase(it, entries.end());  // 해당 ID를 가진 벡터 삭제
            return true;
        }
    }
    return false;
}

bool SphericalGrid::Save() {
    std::ofstream outFile(DATA_FILENAME, std::ios::binary);
    if (!outFile) return false;

    int totalEntries = GetCount();
    outFile.write(reinterpret_cast<const char*>(&totalEntries), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&nextID), sizeof(int));  // ✅ nextID 값 저장

    for (const auto& [sector, entries] : sectorBuckets) {
        for (const auto& entry : entries) {
            outFile.write(reinterpret_cast<const char*>(&entry.id), sizeof(int));
            outFile.write(reinterpret_cast<const char*>(entry.vector), sizeof(float) * vectorDim);
            outFile.write(entry.filePath, MAX_FILE_PATH);
        }
    }

    outFile.close();
    return true;
}

bool SphericalGrid::Load() {
    std::ifstream inFile(DATA_FILENAME, std::ios::binary);
    if (!inFile) return false;

    int totalEntries;
    inFile.read(reinterpret_cast<char*>(&totalEntries), sizeof(int));
    inFile.read(reinterpret_cast<char*>(&nextID), sizeof(int));  // ✅ nextID 값 복원

    for (int i = 0; i < totalEntries; i++) {
        int id;
        std::vector<float> vec(vectorDim);
        char filePath[MAX_FILE_PATH];

        inFile.read(reinterpret_cast<char*>(&id), sizeof(int));
        inFile.read(reinterpret_cast<char*>(vec.data()), sizeof(float) * vectorDim);
        inFile.read(filePath, MAX_FILE_PATH);

        Add(vec, filePath);
    }

    inFile.close();
    return true;
}

size_t SphericalGrid::GetCount() {
    size_t count = 0;
    for (const auto& [sector, entries] : sectorBuckets) {
        count += entries.size();
    }
    return count;
}

// 테스트 코드
int test_mean() {
    constexpr int rows = 16;
    constexpr int cols = 8;

    // 2D 벡터 생성
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols, 1.0f));  // 모든 값 1.0f로 초기화

    // 평균 계산
    std::vector<float> result = MeanVector(matrix);

    // 결과 출력
    for (float val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    return 0;
}

constexpr int VECTOR_DIM = 2048; // 벡터 차원
int test_spherical_grid() {
    try {
        SphericalGrid sdb(VECTOR_DIM);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        // 데이터 입력
        for (int i = 0; i < 10000; ++i) {
            std::vector<float> vec;
            for (int j = 0; j < VECTOR_DIM; ++j) {
                vec.push_back(dis(gen));
            }
            sdb.Add(vec, "C:\\data\\file1.txt");
        }
        // sdb 저장 개수
		printf("SDB Count: %d\n", sdb.GetCount());  

        // 삭제
        sdb.Delete(1);

        // 쿼리 생성
        std::vector<float> query;
        for (int j = 0; j < VECTOR_DIM; ++j) {
            query.push_back(dis(gen));
        }

        // 결과 변수를 올바른 타입으로 선언
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<const WeightEntry*, float>> results;

        for (int i = 0; i < 10; i++) {
            results = sdb.FindNearest(query, 2);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Normal search Elapsed time: " << elapsed.count() << "s\n";
        // --------------------
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; i++) {
            results = sdb.FindNearestFull(query, 2);
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Normal search Elapsed time: " << elapsed.count() << "s\n";

        // 검색 결과
        results = sdb.FindNearest(query, 5);

        std::cout << "Found " << results.size() << " nearest weights:\n";
        for (const auto& result : results) {
            std::cout << "ID: " << result.first->id
                << ", Similarity: " << result.second
                << ", File: " << result.first->filePath << "\n";
        }

        results = sdb.FindNearestFull(query, 5);

        std::cout << "Found " << results.size() << " nearest weights:\n";
        for (const auto& result : results) {
            std::cout << "ID: " << result.first->id
                << ", Similarity: " << result.second
                << ", File: " << result.first->filePath << "\n";
        }
    }
    catch (const std::length_error& e) {
        std::cerr << "Length error: " << e.what() << "\n"; // 크기 관련 예외 처리
    }

    return 0;
}