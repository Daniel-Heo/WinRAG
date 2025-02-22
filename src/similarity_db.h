#pragma once
#define NOMINMAX  // min, max 매크로 충돌 방지 (Windows 환경)
#include <windows.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <memory>      // aligned_alloc
#include <functional>
#include "thread_pool.h"

#ifdef __AVX2__
#define SIMD_TYPE 1 // 0: SSE2 사용,  1: AVX2+FMA3 사용
#else
#define SIMD_TYPE 0 
#endif

#if SIMD_TYPE == 1
#include <immintrin.h>  // AVX, AVX2, FMA3, SSE 관련 헤더
#else
#include <emmintrin.h> // SSE2
#endif

#define INDEX_FILENAME  "sdb.index"
#define LINK_FILENAME   "sdb.data"
#define MAX_FILE_PATH   256

void NormalizeVector(float* vec, size_t size); // 벡터 노멀라이즈
std::vector<float> MeanVector(std::vector<std::vector<float>>& matrix); // 평균 벡터 계산
float CosineSimilarity(const float* v1, const float* v2, size_t size); // 코사인 유사도 계산

int test_similarity_db();
int test_mean();

/****************************************************************
* Class Name: WeightEntry
* Description: 가중치 벡터와 관련 데이터를 저장하는 구조체로, 16바이트 정렬된 메모리를 사용하며 이동 연산만 지원
* Date: 2025-02-21
****************************************************************/
struct WeightEntry {
    float* vector;     // 16바이트 정렬된 가중치 벡터 데이터
    size_t vecSize;    // 벡터의 크기 (차원 수)
    int id;            // 고유 식별자
    char filePath[256]; // 최대 256자의 파일 경로

    // 기본 생성자: 지정된 크기로 벡터를 초기화하고 filePath를 0으로 채움
    WeightEntry(size_t size = 0) : vector(nullptr), vecSize(size), id(0) {
        if (size > 0) {
            vector = static_cast<float*>(_aligned_malloc(size * sizeof(float), 16));
            if (!vector) throw std::bad_alloc(); // 메모리 할당 실패 시 예외 발생
        }
        memset(filePath, 0, sizeof(filePath)); // filePath를 안전하게 초기화
    }

    // 소멸자: 동적으로 할당된 벡터 메모리를 해제
    ~WeightEntry() { if (vector) _aligned_free(vector); }

    // 복사 생성자 삭제: 복사를 방지하여 의도치 않은 메모리 복제를 막음
    WeightEntry(const WeightEntry&) = delete;

    // 이동 생성자: 리소스를 안전하게 이동하며 원본을 초기화
    WeightEntry(WeightEntry&& other) noexcept : vector(other.vector), vecSize(other.vecSize), id(other.id) {
        if (other.filePath[0] != '\0') {
            memcpy(filePath, other.filePath, sizeof(filePath)); // 파일 경로 복사
        }
        else {
            memset(filePath, 0, sizeof(filePath)); // 빈 경로로 초기화
        }
        other.vector = nullptr; // 원본 포인터를 무효화
        other.vecSize = 0;
        memset(other.filePath, 0, sizeof(filePath)); // 원본 경로 초기화
    }

    // 이동 대입 연산자: 기존 리소스를 해제하고 새 리소스를 이동
    WeightEntry& operator=(WeightEntry&& other) noexcept {
        if (this != &other) {
            if (vector) _aligned_free(vector); // 기존 메모리 해제
            vector = other.vector;
            vecSize = other.vecSize;
            id = other.id;
            if (other.filePath[0] != '\0') {
                memcpy(filePath, other.filePath, sizeof(filePath)); // 파일 경로 이동
            }
            else {
                memset(filePath, 0, sizeof(filePath));
            }
            other.vector = nullptr; // 원본 초기화
            other.vecSize = 0;
            memset(other.filePath, 0, sizeof(filePath));
        }
        return *this;
    }
};

/**
 * LSH 기반 k-NN 검색을 지원하는 유사도 데이터베이스
 */
class SimilarityDB {
private:
    std::vector<WeightEntry> weights;        // 저장된 가중치 리스트
    std::vector<int> weightsIndex;           // 가중치 인덱스 리스트
    int vectorDim;                           // 벡터 차원 수
    int numHashTables;                       // 해시 테이블 개수
    int hashSize;                            // 해시 비트 수
    std::vector<std::vector<float>> randomPlanes; // 랜덤 초평면
    std::vector<std::unordered_map<uint32_t, std::vector<int>>> hashTables; // LSH 해시 테이블
    ThreadPool threadPool;                   // 병렬 검색을 위한 스레드 풀
    const std::string INDEX_FILE;            // 인덱스 파일 경로
    const std::string LINK_FILE;             // 링크 파일 경로

    void GenerateRandomPlanes();             // 랜덤 초평면 생성
    uint32_t GetHash(const float* vec, int tableIdx); // 벡터의 LSH 해시값 계산
    void IndexVector(int idx);               // 가중치 벡터를 해시 테이블에 추가
    void SearchTable(int tableIdx, const float* query, std::unordered_set<int>& candidates, std::mutex& mutex);

public:
    // explicit SimilarityDB(int dimension, int tables = 5, int bits = 8); // 생성자
    SimilarityDB(int dimension, int tables = 5, int bits = 8);
    bool Add(const std::vector<float>& vec, const char* filePath); // 가중치 벡터 추가
    bool Delete(int id);                   // 가중치 삭제
    std::vector<std::pair<WeightEntry, float>> FindNearest(const std::vector<float>& queryVec, int k); // k-NN 검색
    bool Sync();                            // 가중치 데이터 파일 저장
    bool Load();                            // 가중치 데이터 파일 불러오기
    size_t GetCount() const;                // 저장된 가중치 수 반환
};