/*******************************************************************************
    파   일   명 : similarity_db.cpp
    프로그램명칭 :  유사도 DB
    작   성   일 : 2025.2.22
    작   성   자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    프로그램용도 : Random Projection 기반 LSH(Locality Sensitive Hashing)를 사용하여 코사인 유사도에 적합한 인덱싱을 구현
                           검색 시 동일한 버킷 내에서만 코사인 유사도를 계산하여 효율성을 높인다.
    참 고 사 항  :
    라 이 센 스  : MIT License

    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  최초 생성
    ----------------------------------------------------------------------------

*******************************************************************************/
// 
// LSH 구현 세부사항 :
// 
// 1. 랜덤 초평면 :
//   GenerateRandomPlanes: 정규 분포를 사용하여 랜덤 초평면을 생성하고 노멀라이즈합니다.
//   각 해시 테이블마다 hashSize 개의 초평면을 사용합니다.
// 2. 해시 함수 :
//   GetHash: 벡터와 초평면의 내적을 계산하여 양수면 1, 음수면 0으로 비트를 설정합니다.
//   결과는 hashSize 비트로 표현되는 uint32_t 해시 값입니다.
// 3. 인덱싱 :
//    IndexVector : 새로 추가된 벡터를 모든 해시 테이블에 매핑합니다.
//    unordered_map을 사용하여 해시 값에 해당하는 벡터 ID 목록을 저장합니다.
// 4. 검색 :
//    쿼리 벡터를 해시하여 동일한 버킷에 속한 후보 벡터를 수집합니다.
//    후보들에 대해서만 코사인 유사도를 계산하여 k개를 반환합니다.
// 5. 튜닝 파라미터 :
//   numHashTables: 해시 테이블 수(기본값 5).많을수록 정확도 증가, 메모리 증가.
//   hashSize : 해시 비트 수(기본값 8).적을수록 충돌 증가, 많을수록 계산 비용 증가.
// 7. 장점 :
//    전체 벡터를 순회하지 않고 후보군만 검사하므로 성능이 개선됩니다.
//    데이터 크기가 클수록 LSH의 이점이 두드러집니다.
// 8. 한계 :
//    정확도는 numHashTables와 hashSize에 의존하며, 완벽한 k - NN이 아닌 근사 결과 제공.
//    소규모 데이터에서는 오버헤드가 클 수 있음.
//
// 적용된 최적화 상세 :
// 메모리 정렬 :
//   WeightEntry의 vector를 std::vector<float>에서 float* 로 변경하고, _aligned_malloc으로 16바이트 정렬된 메모리를 할당.
//   _mm_load_ps와 _mm_store_ps를 사용하여 정렬된 메모리에서 SIMD 연산 수행.
//   메모리 해제는 _aligned_free로 처리.
//   이동 생성자와 이동 대입 연산자를 추가하여 안전한 리소스 관리.
// 스레드 풀 :
//   ThreadPool 클래스를 구현하여 고정된 수의 스레드를 생성(CPU 코어 수 기반).
//   작업 큐에 검색 작업을 추가하고, 스레드가 작업을 처리하도록 함.
//   FindNearest에서 각 해시 테이블 검색을 스레드 풀에 위임.
//   스레드 간 동기화를 위해 std::mutex 사용.
// 주의사항 :
//     컴파일 : Visual Studio에서 _aligned_malloc과 SSE를 사용하므로 / arch : SSE2 옵션을 활성화해야 합니다.
//     메모리 관리 : WeightEntry의 소멸자에서 메모리를 해제하므로 중복 해제 주의.
//     스레드 풀 종료 : 프로그램 종료 시 스레드 풀이 안전하게 종료됨.
//     성능 개선 효과 :
// 메모리 정렬 : SIMD 연산이 더 효율적으로 실행되며, 캐시 라인 활용도가 높아짐.
// 스레드 풀 : 스레드 생성 / 소멸 오버헤드가 없어지고, 작업 분배가 안정적.

#include "similarity_db.h"

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

/****************************************************************
* Function Name: SimilarityDB
* Description: SimilarityDB 객체를 초기화하고 해시 테이블 및 초평면 설정
* Parameters:
*   - dimension: 벡터 차원 수
*   - tables: 해시 테이블 수 (기본값 5)
*   - bits: 해시 비트 수 (기본값 8)
* Return: 없음 (생성자)
* Date: 2025-02-21
****************************************************************/
SimilarityDB::SimilarityDB(int dimension, int tables, int bits)
    : vectorDim(dimension), numHashTables(tables), hashSize(bits),
    threadPool(std::thread::hardware_concurrency()) {
    if (tables <= 0 || bits <= 0 || dimension <= 0) {
        throw std::invalid_argument("Invalid SimilarityDB parameters");
    }
    if (tables > static_cast<int>(hashTables.max_size())) {
        std::cerr << "Error: numHashTables exceeds max_size: " << hashTables.max_size() << "\n";
        throw std::length_error("Too many hash tables");
    }
    hashTables.resize(numHashTables); // 해시 테이블 크기 설정
    GenerateRandomPlanes();           // 초평면 생성
}


/****************************************************************
* Function Name: GenerateRandomPlanes
* Description: LSH용 랜덤 초평면을 생성하고 노멀라이즈
* Parameters: 없음
* Return: 없음
* Date: 2025-02-21
****************************************************************/
void SimilarityDB::GenerateRandomPlanes() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    size_t totalPlanes = numHashTables * hashSize;
    if (totalPlanes > randomPlanes.max_size()) {
        std::cerr << "Error: randomPlanes size exceeds max_size: " << randomPlanes.max_size() << "\n";
        throw std::length_error("randomPlanes too large");
    }
    randomPlanes.resize(totalPlanes); // 초평면 배열 크기 조정
    std::cout << "Generating " << totalPlanes << " random planes\n";

    for (auto& plane : randomPlanes) {
        if (vectorDim > plane.max_size()) {
            std::cerr << "Error: vectorDim exceeds plane max_size: " << plane.max_size() << "\n";
            throw std::length_error("plane vector too large");
        }
        plane.resize(vectorDim); // 각 초평면의 차원 설정
        for (float& v : plane) v = dist(gen); // 정규 분포로 값 생성
        NormalizeVector(plane.data(), vectorDim); // 초평면 노멀라이즈
    }
}

/****************************************************************
* Function Name: GetHash
* Description: 주어진 벡터에 대해 LSH 해시 값을 계산
* Parameters:
*   - vec: 해시를 계산할 벡터
*   - tableIdx: 사용할 해시 테이블 인덱스
* Return: 계산된 해시 값 (uint32_t)
* Date: 2025-02-21
****************************************************************/
uint32_t  SimilarityDB::GetHash(const float* vec, int tableIdx) {
    if (tableIdx < 0 || tableIdx >= numHashTables) {
        std::cerr << "Error: Invalid tableIdx: " << tableIdx << "\n";
        return 0; // 오류 시 기본값 반환
    }
    uint32_t hash = 0;
    for (int i = 0; i < hashSize; i++) {
        size_t planeIdx = tableIdx * hashSize + i;
        if (planeIdx >= randomPlanes.size()) {
            std::cerr << "Error: planeIdx out of bounds: " << planeIdx << "\n";
            return hash; // 범위 초과 시 중단
        }
        const auto& plane = randomPlanes[planeIdx];
        float dot = CosineSimilarity(vec, plane.data(), vectorDim);
        if (dot >= 0) hash |= (1 << i); // 양수면 비트 설정
    }
    return hash;
}

/****************************************************************
* Function Name: IndexVector
* Description: 주어진 가중치 인덱스를 모든 해시 테이블에 추가
* Parameters:
*   - idx: 인덱스할 가중치의 위치
* Return: 없음
* Date: 2025-02-21
****************************************************************/
void  SimilarityDB::IndexVector(int idx) {
    if (idx < 0 || static_cast<size_t>(idx) >= weights.size()) {
        std::cerr << "Error: Invalid idx: " << idx << "\n";
        return;
    }
    for (int t = 0; t < numHashTables; t++) {
        if (t >= static_cast<int>(hashTables.size())) {
            std::cerr << "Error: hashTables index out of bounds: " << t << "\n";
            continue;
        }
        uint32_t hash = GetHash(weights[idx].vector, t);
        hashTables[t][hash].push_back(idx); // 해시 버킷에 인덱스 추가
    }
}

/****************************************************************
* Function Name: SearchTable
* Description: 지정된 해시 테이블에서 쿼리와 동일한 해시 값을 가진 후보를 찾음
* Parameters:
*   - tableIdx: 검색할 해시 테이블 인덱스
*   - query: 검색 쿼리 벡터
*   - candidates: 후보 인덱스 집합 (참조)
*   - mutex: 동기화를 위한 뮤텍스 (참조)
* Return: 없음
* Date: 2025-02-21
****************************************************************/
void  SimilarityDB::SearchTable(int tableIdx, const float* query, std::unordered_set<int>& candidates,
    std::mutex& mutex) {
    if (tableIdx < 0 || tableIdx >= static_cast<int>(hashTables.size())) {
        std::cerr << "Error: Invalid tableIdx in SearchTable: " << tableIdx << "\n";
        return;
    }
    uint32_t hash = GetHash(query, tableIdx);
    auto it = hashTables[tableIdx].find(hash);
    if (it != hashTables[tableIdx].end()) {
        std::lock_guard<std::mutex> lock(mutex); // 동기화
        for (int idx : it->second) {
            candidates.insert(idx); // 동일 해시 버킷의 인덱스 추가
        }
    }
}

/****************************************************************
* Function Name: Add
* Description: 새 가중치 벡터와 파일 경로를 추가하고 인덱싱
* Parameters:
*   - vec: 추가할 가중치 벡터
*   - filePath: 관련 파일 경로
* Return: 성공 시 true, 실패 시 false
* Date: 2025-02-21
****************************************************************/
bool  SimilarityDB::Add(const std::vector<float>& vec, const char* filePath) {
    if (vec.size() != vectorDim || strlen(filePath) >= MAX_FILE_PATH) return false;

    WeightEntry entry(vectorDim);
    if (!entry.vector) return false;

    memcpy(entry.vector, vec.data(), vectorDim * sizeof(float));
    NormalizeVector(entry.vector, vectorDim); // 벡터 정규화
    entry.id = weights.empty() ? 0 : weights.back().id + 1; // 새로운 ID 할당
    strncpy_s(entry.filePath, filePath, MAX_FILE_PATH);

    weights.push_back(std::move(entry));
    weightsIndex.push_back(static_cast<int>(weights.size() - 1)); // weights의 인덱스를 weightsIndex에 저장

    IndexVector(weightsIndex.back()); // 새로 추가된 벡터를 해시 테이블에 인덱싱
    return true;
}

/****************************************************************
* Function Name: Delete
* Description: 지정된 ID의 가중치를 삭제하고 인덱스를 업데이트
* Parameters:
*   - id: 삭제할 가중치의 ID
* Return: 성공 시 true, 실패 시 false
* Date: 2025-02-22
****************************************************************/
bool  SimilarityDB::Delete(int id) {
    auto it = std::find(weightsIndex.begin(), weightsIndex.end(), id);
    if (it == weightsIndex.end()) {
        std::cerr << "Error: Weight with ID " << id << " not found\n";
        return false;
    }

    int removeIndex = static_cast<int>(std::distance(weightsIndex.begin(), it)); // weightsIndex 내에서 삭제할 위치 찾기
    int actualIndex = *it; // weights에서 삭제할 실제 인덱스

    // weights에서 해당 가중치 제거
    weights.erase(weights.begin() + actualIndex);

    // weightsIndex에서 제거
    weightsIndex.erase(it);

    // 삭제한 인덱스 이후의 값들을 -1씩 조정
    for (size_t i = removeIndex; i < weightsIndex.size(); ++i) {
        weightsIndex[i]--; // -1 감소
    }

    // 해시 테이블에서도 삭제
    for (int t = 0; t < numHashTables; t++) {
        for (auto& bucket : hashTables[t]) {
            auto& indices = bucket.second;
            indices.erase(std::remove(indices.begin(), indices.end(), actualIndex), indices.end());
        }
    }

    std::cout << "Successfully deleted weight with ID " << id << "\n";
    return true;
}

/****************************************************************
    * Function Name: FindNearest
    * Description: 쿼리 벡터에 대해 k개의 가장 가까운 가중치를 코사인 유사도로 검색
    *              k보다 적은 수의 이웃이 검색되면 검색된 수만큼 반환, 없으면 빈 벡터 반환
    * Parameters:
    *   - queryVec: 검색 쿼리 벡터
    *   - k: 반환할 최대 이웃 수
    * Return: 가중치와 유사도 쌍의 벡터 (검색된 수에 따라 크기 변동, 없으면 size() == 0)
    * Author: Grok 3 by xAI
    * Date: 2025-02-21
    ****************************************************************/
std::vector<std::pair<WeightEntry, float>>  SimilarityDB::FindNearest(const std::vector<float>& queryVec, int k) {
    if (queryVec.size() != vectorDim || weights.empty() || weightsIndex.empty()) {
        return {}; // 가중치가 없으면 빈 결과 반환
    }

    // 쿼리 벡터 정렬된 메모리 할당
    float* normalizedQuery = static_cast<float*>(_aligned_malloc(vectorDim * sizeof(float), 16));
    if (!normalizedQuery) return {};
    memcpy(normalizedQuery, queryVec.data(), vectorDim * sizeof(float));
    NormalizeVector(normalizedQuery, vectorDim);

    std::unordered_set<int> candidates;
    std::mutex candidatesMutex;

    std::condition_variable cv;
    std::mutex cvMutex;
    int threadsFinished = 0;  // 완료된 스레드 개수

    for (int t = 0; t < numHashTables; t++) {
        threadPool.Enqueue([this, t, normalizedQuery, &candidates, &candidatesMutex, &cv, &cvMutex, &threadsFinished]() {
            SearchTable(t, normalizedQuery, candidates, candidatesMutex);

            // 스레드 완료 후 알림
            std::lock_guard<std::mutex> lock(cvMutex);
            threadsFinished++;
            cv.notify_one();  // 대기 중인 스레드 깨우기
            });
    }

    // 모든 스레드가 끝날 때까지 대기
    std::unique_lock<std::mutex> lock(cvMutex);
    cv.wait(lock, [&threadsFinished, this]() { return threadsFinished == numHashTables; });

    if (candidates.empty()) {
        _aligned_free(normalizedQuery);
        return {};
    }

    std::vector<std::pair<float, WeightEntry>> similarities;
    similarities.reserve(candidates.size());

    for (int id : weightsIndex) {
        if (id < 0 || id >= static_cast<int>(weightsIndex.size())) {
            std::cerr << "Error: Candidate ID " << id << " is invalid (out of range)\n";
            continue;
        }

        float sim = CosineSimilarity(normalizedQuery, weights[id].vector, vectorDim);
        similarities.emplace_back(sim, WeightEntry(vectorDim));
        memcpy(similarities.back().second.vector, weights[id].vector, vectorDim * sizeof(float));
        similarities.back().second.id = weights[id].id;
        strncpy_s(similarities.back().second.filePath, weights[id].filePath, MAX_FILE_PATH);
    }

    if (similarities.empty()) {
        _aligned_free(normalizedQuery);
        return {};
    }

    size_t actualK = std::min(static_cast<size_t>(k), similarities.size());
    std::partial_sort(similarities.begin(),
        similarities.begin() + actualK,
        similarities.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<std::pair<WeightEntry, float>> results;
    results.reserve(actualK);
    for (size_t i = 0; i < actualK; i++) {
        results.emplace_back(std::move(similarities[i].second), similarities[i].first);
    }

    _aligned_free(normalizedQuery);
    return results;
}

/****************************************************************
* Function Name: Sync
* Description: 현재 가중치와 파일 경로를 파일에 저장
* Parameters: 없음
* Return: 성공 시 true, 실패 시 false
* Date: 2025-02-21
****************************************************************/
bool  SimilarityDB::Sync() {
    std::ofstream indexFile(INDEX_FILE, std::ios::binary);
    std::ofstream linkFile(LINK_FILE, std::ios::binary);
    if (!indexFile.is_open() || !linkFile.is_open()) return false;

    int count = static_cast<int>(weights.size());
    indexFile.write(reinterpret_cast<const char*>(&count), sizeof(int));
    linkFile.write(reinterpret_cast<const char*>(&count), sizeof(int));

    for (const auto& entry : weights) {
        int16_t vecSize = static_cast<int16_t>(entry.vecSize);
        indexFile.write(reinterpret_cast<const char*>(&entry.id), sizeof(int));
        indexFile.write(reinterpret_cast<const char*>(&vecSize), sizeof(int16_t));
        indexFile.write(reinterpret_cast<const char*>(entry.vector), vecSize * sizeof(float));
        linkFile.write(reinterpret_cast<const char*>(&entry.id), sizeof(int));
        linkFile.write(entry.filePath, MAX_FILE_PATH);
    }

    return true;
}

/****************************************************************
* Function Name: Load
* Description: 파일에서 가중치와 파일 경로를 로드하여 인덱스 복원
* Parameters: 없음
* Return: 성공 시 true, 실패 시 false
* Date: 2025-02-21
****************************************************************/
bool  SimilarityDB::Load() {
    std::ifstream indexFile(INDEX_FILE, std::ios::binary);
    std::ifstream linkFile(LINK_FILE, std::ios::binary);
    if (!indexFile.is_open() || !linkFile.is_open()) return false;

    weights.clear();
    for (auto& table : hashTables) table.clear(); // 기존 해시 테이블 초기화

    int count;
    indexFile.read(reinterpret_cast<char*>(&count), sizeof(int));
    linkFile.read(reinterpret_cast<char*>(&count), sizeof(int));
    if (count < 0 || static_cast<size_t>(count) > weights.max_size()) {
        std::cerr << "Error: Invalid count from file: " << count << ", max_size: " << weights.max_size() << "\n";
        return false;
    }
    weights.reserve(count); // 메모리 예약

    for (int i = 0; i < count; i++) {
        WeightEntry entry(vectorDim);
        int16_t vecSize;
        indexFile.read(reinterpret_cast<char*>(&entry.id), sizeof(int));
        indexFile.read(reinterpret_cast<char*>(&vecSize), sizeof(int16_t));
        entry.vecSize = vecSize;
        indexFile.read(reinterpret_cast<char*>(entry.vector), vecSize * sizeof(float));
        linkFile.read(reinterpret_cast<char*>(&entry.id), sizeof(int));
        linkFile.read(entry.filePath, MAX_FILE_PATH);

        weights.push_back(std::move(entry));
        IndexVector(entry.id); // 로드된 엔트리 인덱싱
    }

    return true;
}

/****************************************************************
* Function Name: GetCount
* Description: 저장된 가중치 수 반환
* Parameters: 없음
* Return: 가중치 수 (size_t)
* Date: 2025-02-21
****************************************************************/
size_t  SimilarityDB::GetCount() const { return weights.size(); }

/*
* 
해시 테이블의 개수와 비트 수의 의미:
1. 해시 테이블의 개수의 의미:
LSH의 기본 개념: LSH는 고차원 데이터를 효율적으로 검색하기 위해 해시 함수를 사용하여 데이터를 "버킷(bucket)"에 그룹화합니다. 비슷한 데이터는 동일한 해시 값(버킷)에 매핑될 가능성이 높습니다.
여러 해시 테이블 사용 이유: 단일 해시 테이블만 사용하면 해시 충돌(다른 데이터가 같은 버킷에 매핑됨)이 많아지거나, 일부 가까운 데이터가 다른 버킷에 떨어질 수 있습니다. 이를 보완하기 위해 여러 개의 독립적인 해시 테이블을 사용하여 검색 정확도를 높입니다.
5개의 해시 테이블: 코드에서는 5개의 서로 다른 해시 테이블을 생성합니다. 각 테이블은 독립적인 해시 함수(랜덤 초평면 집합)를 사용하여 데이터를 매핑합니다. 검색 시 이 5개 테이블을 모두 확인하여 더 많은 후보를 수집합니다.

코드에서의 역할:
hashTables는 std::vector<std::unordered_map<uint32_t, std::vector<int>>>로 정의되며, 크기가 numHashTables (여기서는 5)로 초기화됩니다.
각 hashTables[i]는 특정 해시 함수로 계산된 키(uint32_t)와 해당 키에 속하는 가중치 인덱스 목록(std::vector<int>)을 저장합니다.
예: IndexVector에서 각 가중치를 5개의 해시 테이블에 추가하고, FindNearest에서 5개 테이블을 병렬로 검색합니다.
영향:
장점: 테이블 수가 많아질수록 가까운 이웃을 놓칠 확률(漏れ)이 줄어듭니다 (정확도 향상).
단점: 메모리 사용량과 검색 시간이 증가합니다.
5라는 값: 테스트 용도로 적당한 값으로 설정된 예시입니다. 실제로는 데이터 크기와 요구 정확도에 따라 조정 가능합니다.

2. 해시 비트 수의 의미:
LSH 해시 함수: 코드에서는 Random Projection 기반 LSH를 사용합니다. 각 해시 테이블은 hashSize 개의 랜덤 초평면(random planes)을 사용하여 해시 값을 생성합니다.
8비트: 하나의 해시 값은 8비트로 표현되며, 이는 최대 2^8 = 256개의 서로 다른 해시 버킷을 생성할 수 있음을 의미합니다.
해시 값 계산: GetHash 함수에서 벡터와 8개의 초평면의 내적을 계산하고, 결과가 양수면 해당 비트를 1로, 음수면 0으로 설정합니다. 예:
초평면 8개와의 내적 결과: [+,-,+,-,-,+,+,+]
해시 값: 10100111 (2진수, 8비트)

코드에서의 역할:
randomPlanes는 numHashTables * hashSize (5 * 8 = 40)개의 초평면을 생성합니다.
GetHash는 주어진 벡터에 대해 8회의 내적을 계산하여 8비트 해시 값을 만듭니다.
hashTables[t][hash]의 키(hash)는 uint32_t로 저장되지만, 실질적으로 하위 8비트만 사용됩니다 (최대 256개 버킷).
영향:
장점: 비트 수가 많을수록 해시 값의 해상도가 높아져 비슷한 벡터가 같은 버킷에 매핑될 확률이 증가합니다 (정확도 향상).
단점: 비트 수가 많아지면 버킷 수가 기하급수적으로 증가하여 각 버킷에 속하는 데이터가 적어질 수 있고, 계산 비용이 증가합니다.
8라는 값: 적당한 해상도와 계산 비용의 균형을 맞춘 예시 값입니다. 데이터 분포에 따라 조정 가능합니다.
*/
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

int test_similarity_db() {
    try {
        SimilarityDB sdb(3, 5, 8); // 3차원 벡터, 5개 해시 테이블, 8비트 해시

        //if (sdb.Load()) std::cout << "Loaded existing db\n"; // 기존 인덱스 로드 시도

        // 테스트 데이터 추가
        sdb.Add({ 0.1f, 2.0f, 0.1f }, "C:\\data\\file1.txt");
        sdb.Add({ 4.0f, 5.0f, 6.0f }, "C:\\data\\file2.txt");
        sdb.Add({ 1.5f, 2.5f, 3.5f }, "C:\\data\\file3.txt");

        // 삭제 테스트
        sdb.Delete(1); // ID 1인 가중치 삭제

        std::vector<float> query = { 1.2f, 2.2f, 3.2f };
        // 시간 계산
		auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<WeightEntry, float>> results;
		for ( int i = 0; i < 100000; i++) {
			 results = sdb.FindNearest(query, 2); // 2개의 최근접 이웃 검색
		}
        //auto results = sdb.FindNearest(query, 2); // 2개의 최근접 이웃 검색
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		std::cout << "Elapsed time: " << elapsed.count() << "s\n";

        std::cout << "Found " << results.size() << " nearest weights:\n";
        for (const auto& result : results) {
            std::cout << "ID: " << result.first.id
                << ", Similarity: " << result.second
                << ", File: " << result.first.filePath << "\n";
        }

        if (sdb.Sync()) std::cout << "db saved successfully\n"; // 인덱스 저장
    }
    catch (const std::length_error& e) {
        std::cerr << "Length error: " << e.what() << "\n"; // 크기 관련 예외 처리
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n"; // 기타 예외 처리
    }

    return 0;
}