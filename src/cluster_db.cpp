/*******************************************************************************
    파   일   명 : spherical_grid.cpp
    프로그램명칭 :  유사도 DB
    작   성   일 : 2025.2.22
    작   성   자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    프로그램용도 : 클러스터링을 사용하여 효율성을 높인다.
                           
	참 고 사 항  : 정확한 연산이 필요한 경우 FindNearestFull을 사용하고 10000개 이상의 데이터에서 빠른 근사치 계산을 원할 경우 FindNearestCluster를 사용한다.
    라 이 센 스  : MIT License

    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.23   Daniel Heo  최초 생성
    ----------------------------------------------------------------------------

*******************************************************************************/
#include "cluster_db.h"

// RunKMeansClustering에서 사용하는 SIMD로 평균 벡터 계산
//   행방향 처리 ( 메모리 연속성 높여서 처리 속도 향상 ) - 차원수가 많을수록 효과적
// SIMD로 평균 벡터 계산 (SIMD 미사용 환경도 지원)
std::vector<float> MeanVectorSIMD(const std::vector<WeightEntry>& entries, size_t vectorDim) {
    size_t entryCount = entries.size();
    std::vector<float> mean(vectorDim, 0.0f);

    if (entryCount == 0) return mean;

#if SIMD_TYPE == 1  // AVX2 코드 (256비트, 8 float씩 처리)
    size_t simdWidth = 8;
    size_t simdEnd = vectorDim - (vectorDim % simdWidth);

    for (size_t i = 0; i < simdEnd; i += simdWidth) {
        __m256 sum = _mm256_setzero_ps();

        for (const auto& entry : entries) {
            sum = _mm256_add_ps(sum, _mm256_loadu_ps(entry.vector + i));
        }

        sum = _mm256_div_ps(sum, _mm256_set1_ps(static_cast<float>(entryCount)));
        _mm256_storeu_ps(mean.data() + i, sum);
    }

    // 남은 원소 처리
    for (size_t i = simdEnd; i < vectorDim; ++i) {
        float sum = 0.0f;
        for (const auto& entry : entries) {
            sum += entry.vector[i];
        }
        mean[i] = sum / entryCount;
    }

#elif SIMD_TYPE == 0  // SSE2 코드 (128비트, 4 float씩 처리)
    size_t simdWidth = 4;
    size_t simdEnd = vectorDim - (vectorDim % simdWidth);

    for (size_t i = 0; i < simdEnd; i += simdWidth) {
        __m128 sum = _mm_setzero_ps();

        for (const auto& entry : entries) {
            sum = _mm_add_ps(sum, _mm_loadu_ps(entry.vector + i));
        }

        sum = _mm_div_ps(sum, _mm_set1_ps(static_cast<float>(entryCount)));
        _mm_storeu_ps(mean.data() + i, sum);
    }

    // 남은 원소 처리
    for (size_t i = simdEnd; i < vectorDim; ++i) {
        float sum = 0.0f;
        for (const auto& entry : entries) {
            sum += entry.vector[i];
        }
        mean[i] = sum / entryCount;
    }

#else  // SIMD 사용 안 함 (범용코드, 언제나 동작)
    // SIMD_TYPE이 AVX2나 SSE2가 아닌 경우 일반적인 방식으로 계산
    for (size_t i = 0; i < vectorDim; ++i) {
        float sum = 0.0f;
        for (const auto& entry : entries) {
            sum += entry.vector[i];
        }
        mean[i] = sum / entryCount;
    }
#endif

    return mean;
}

// 클러스터DB 생성자
ClusterDB::ClusterDB(int dimension)
    : vectorDim(dimension), numClusters(CLUSTER_COUNT), isDataChanged(false), currentId(0) {
    clusters.resize(numClusters);
    for (auto& cluster : clusters) {
        cluster.centroid.resize(vectorDim, 0.0f);
    }
}

// 데이터 추가
bool ClusterDB::Add(const std::vector<float>& vec, const char* filePath) {
    if (vec.size() != vectorDim) return false;

    WeightEntry entry(vectorDim);
    memcpy(entry.vector, vec.data(), vectorDim * sizeof(float));
    NormalizeVector(entry.vector, vectorDim);

    entry.id = currentId++;
    strncpy_s(entry.filePath, filePath, MAX_FILE_PATH);

    // 가까운 클러스터 찾기
    int clusterIndex = GetNearestClusterIndex(entry.vector);
    clusters[clusterIndex].entries.emplace_back(std::move(entry));

	isDataChanged = true;

    return true;
}

// K-means 클러스터링 알고리즘 실행
void ClusterDB::RunKMeansClustering(void) {
    // 초기 centroid 무작위 초기화 (필수!)
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& cluster : clusters) {
        for (auto& val : cluster.centroid) {
            val = dis(gen);
        }
        NormalizeVector(cluster.centroid.data(), vectorDim);
    }

    for (int iter = 0; iter < numClusters; ++iter) {
        // 각 클러스터 중심 SIMD로 빠르게 재계산
        for (auto& cluster : clusters) {
            if (!cluster.entries.empty()) {
                // SIMD 최적화 함수로 평균 계산
                cluster.centroid = MeanVectorSIMD(cluster.entries, vectorDim);
                NormalizeVector(cluster.centroid.data(), vectorDim);
            }
        }

        // 엔트리 재할당
        std::vector<Cluster> newClusters(numClusters);
        for (auto& cluster : newClusters) {
            cluster.centroid.resize(vectorDim, 0.0f);
        }

        for (auto& cluster : clusters) {
            for (auto& entry : cluster.entries) {
                int nearestIndex = GetNearestClusterIndex(entry.vector);
                newClusters[nearestIndex].entries.emplace_back(std::move(entry));
            }
        }
        clusters = std::move(newClusters);
    }
}

// 가장 가까운 클러스터 찾기
int ClusterDB::GetNearestClusterIndex(const float* vec) {
    float maxSim = -1.0f;
    int nearestIndex = 0;
    for (int i = 0; i < numClusters; ++i) {
        float sim = CosineSimilarity(vec, clusters[i].centroid.data(), vectorDim);
        if (sim > maxSim) {
            maxSim = sim;
            nearestIndex = i;
        }
    }
    return nearestIndex;
}

// 클러스터 기반 검색
std::vector<std::pair<const WeightEntry*, float>> ClusterDB::FindNearestCluster(const std::vector<float>& queryVec, int k) {
    if (queryVec.size() != vectorDim) return {};

	// 데이터 변경이 있으면 클러스터링 실행
	if (isDataChanged) {
		RunKMeansClustering();
		isDataChanged = false;
	}

    float* normalizedQuery = static_cast<float*>(_aligned_malloc(vectorDim * sizeof(float), 16));
    memcpy(normalizedQuery, queryVec.data(), vectorDim * sizeof(float));
    NormalizeVector(normalizedQuery, vectorDim);

    int nearestClusterIdx = GetNearestClusterIndex(normalizedQuery);
    auto& entries = clusters[nearestClusterIdx].entries;

    std::vector<std::pair<float, const WeightEntry*>> similarities;
    for (const auto& entry : entries) {
        float sim = CosineSimilarity(normalizedQuery, entry.vector, vectorDim);
        similarities.emplace_back(sim, &entry);
    }

    _aligned_free(normalizedQuery);

    size_t actualK = std::min(static_cast<size_t>(k), similarities.size());
    std::partial_sort(similarities.begin(), similarities.begin() + actualK, similarities.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<std::pair<const WeightEntry*, float>> results(actualK);
    for (size_t i = 0; i < actualK; ++i) {
        results[i] = { similarities[i].second, similarities[i].first };
    }

    return results;
}

// 전체 데이터를 검색하여 유사도가 가장 높은 k개의 벡터를 찾음 (Full Scan 방식)
std::vector<std::pair<const WeightEntry*, float>> ClusterDB::FindNearestFull(const std::vector<float>& queryVec, int k) {
    if (queryVec.size() != vectorDim) return {};

    // 쿼리 벡터를 정규화
    float* normalizedQuery = static_cast<float*>(_aligned_malloc(vectorDim * sizeof(float), 16));
    memcpy(normalizedQuery, queryVec.data(), vectorDim * sizeof(float));
    NormalizeVector(normalizedQuery, vectorDim);

    std::vector<std::pair<float, const WeightEntry*>> similarities;

    // 모든 클러스터의 모든 데이터 순회
    for (const auto& cluster : clusters) {
        for (const auto& entry : cluster.entries) {
            float similarity = CosineSimilarity(normalizedQuery, entry.vector, vectorDim);
            similarities.emplace_back(similarity, &entry);
        }
    }

    _aligned_free(normalizedQuery);

    // 상위 k개 항목만 부분 정렬 (유사도 기준 내림차순)
    size_t actualK = std::min(static_cast<size_t>(k), similarities.size());
    std::partial_sort(similarities.begin(), similarities.begin() + actualK, similarities.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    // 결과 저장 및 반환
    std::vector<std::pair<const WeightEntry*, float>> results;
    results.reserve(actualK);
    for (size_t i = 0; i < actualK; ++i) {
        results.emplace_back(similarities[i].second, similarities[i].first);
    }

    return results;
}

// 전체 클러스터 DB를 파일로 저장하는 함수
bool ClusterDB::Save(const char* filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) return false;

    // 전체 벡터 차원 수 저장
    outFile.write(reinterpret_cast<const char*>(&vectorDim), sizeof(vectorDim));

    // 전체 데이터 개수 얻기
    size_t totalEntries = GetCount();
    outFile.write(reinterpret_cast<const char*>(&totalEntries), sizeof(totalEntries));

    // 클러스터와 관계없이 모든 데이터 저장
    for (const auto& cluster : clusters) {
        for (const auto& entry : cluster.entries) {
            outFile.write(reinterpret_cast<const char*>(&entry.id), sizeof(entry.id));
            outFile.write(reinterpret_cast<const char*>(entry.vector), sizeof(float) * vectorDim);
            outFile.write(entry.filePath, MAX_FILE_PATH);
        }
    }

    outFile.close();
    return true;
}

// 전체 클러스터 DB를 파일에서 불러오는 함수
bool ClusterDB::Load(const char* filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) return false;

    int fileVectorDim = 0;
    size_t totalEntries = 0;

    // 벡터 차원 로드
    inFile.read(reinterpret_cast<char*>(&fileVectorDim), sizeof(fileVectorDim));
    if (fileVectorDim != vectorDim) {
        inFile.close();
        return false;
    }

    // 전체 데이터 개수 로드
    inFile.read(reinterpret_cast<char*>(&totalEntries), sizeof(totalEntries));

    // 기존 데이터 초기화
    clusters.clear();
    clusters.resize(numClusters);
    for (auto& cluster : clusters) {
        cluster.centroid.resize(vectorDim, 0.0f);
    }

    int maxId = currentId;

    // 로드한 데이터를 임시 클러스터(예: clusters[0])에 몰아서 저장
    for (size_t i = 0; i < totalEntries; ++i) {
        WeightEntry entry(vectorDim);
        inFile.read(reinterpret_cast<char*>(&entry.id), sizeof(entry.id));
        inFile.read(reinterpret_cast<char*>(entry.vector), sizeof(float) * vectorDim);
        inFile.read(entry.filePath, MAX_FILE_PATH);

        clusters[0].entries.emplace_back(std::move(entry));
		if (entry.id >= maxId) maxId = entry.id+1;
    }

	currentId = maxId;

    inFile.close();
    return true;
}

// 주어진 ID를 가진 데이터를 삭제하는 함수
bool ClusterDB::Delete(int id) {
    // 모든 클러스터 전체를 돌면서 데이터 삭제
    for (auto& cluster : clusters) {
        auto it = std::remove_if(cluster.entries.begin(), cluster.entries.end(),
            [id](const WeightEntry& entry) {
                return entry.id == id;
            });

        if (it != cluster.entries.end()) {
            cluster.entries.erase(it, cluster.entries.end());
            return true;
        }
    }
    return false; // 데이터 없으면 false 반환
}


// 전체 데이터 개수
size_t ClusterDB::GetCount() {
    size_t cnt = 0;
    for (const auto& c : clusters) cnt += c.entries.size();
    return cnt;
}

int ClusterDB::GetCurrentId() {
	return currentId;
}


// 테스트 코드
int test_mean() {
    constexpr int rows = 16;
    constexpr int cols = 8;

    // 2D 벡터 생성
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols, 2.0f));  // 모든 값 1.0f로 초기화

    // 평균 계산
    std::vector<float> result = MeanVector(matrix);

    // 결과 출력
    for (float val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    return 0;
}

// 성능 테스트
int test_cluster_db() {
    constexpr int VECTOR_DIM = 2048;  // 벡터 차원 설정
    constexpr int DATA_COUNT = 10000; // 데이터 개수

    ClusterDB cdb(VECTOR_DIM);

    // 임의의 데이터 생성
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 시간 측정
    auto start = std::chrono::high_resolution_clock::now();
    //cdb.Load("cluster_db.bin");
    for (int i = 0; i < DATA_COUNT; ++i) {
        std::vector<float> vec(VECTOR_DIM);
        for (auto& val : vec) val = dist(gen);
        cdb.Add(vec, "C:\\test\\file.txt");
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Insert time: " << elapsed.count() << "s\n";

    cdb.Delete(1);
	printf("SDB Count: %d\n", cdb.GetCount());

    // 쿼리 생성 및 검색 수행
    std::vector<float> queryVec(VECTOR_DIM);
    for (auto& val : queryVec) val = dist(gen);

    // 검색
    std::vector<std::pair<const WeightEntry*, float>> results;
    results = cdb.FindNearestCluster(queryVec, 3); // RunKMeansClustering 수행

    // 시간 측정
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; ++i) {
        results = cdb.FindNearestCluster(queryVec, 3);
	}
	end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
	std::cout << "Scan time: " << elapsed.count() << "s\n";

    // 결과 출력
    for (auto& res : results) {
        printf("ID: %d, 유사도: %.3f, 파일경로: %s\n",
            res.first->id, res.second, res.first->filePath);
    }

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; ++i) {
		results = cdb.FindNearestFull(queryVec, 3);
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	std::cout << "Fullscan time: " << elapsed.count() << "s\n";
    

    // 결과 출력
    for (auto& res : results) {
        printf("ID: %d, 유사도: %.3f, 파일경로: %s\n",
            res.first->id, res.second, res.first->filePath);
    }

    // 저장
	cdb.Save("cluster_db.bin");

    return 0;
}