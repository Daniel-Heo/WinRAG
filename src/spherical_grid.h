/*******************************************************************************
    파   일   명 : spherical_grid.h
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

//#define INDEX_FILENAME  "sdb.index"
#define DATA_FILENAME   "sdb.data"
#define MAX_FILE_PATH   256
#define SECTOR_COUNT_PER_PAIR 12 // 각 (x, y) 쌍의 각도를 12개 구역으로 분할

void NormalizeVector(float* vec, size_t size); // 벡터 노멀라이즈
std::vector<float> MeanVector(std::vector<std::vector<float>>& matrix); // 평균 벡터 계산
float CosineSimilarity(const float* v1, const float* v2, size_t size); // 코사인 유사도 계산

int test_spherical_grid();
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
        memcpy(filePath, other.filePath, sizeof(filePath));
        other.vector = nullptr;
        other.vecSize = 0;
        memset(other.filePath, 0, sizeof(filePath));
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
class SphericalGrid {
private:
    int vectorDim;
    int numSectors;
    std::unordered_map<int, std::vector<WeightEntry>> sectorBuckets;
    int nextID = 0;  // 항상 증가하는 ID (삭제 후에도 재사용하지 않음)

    int GetSectorIndex(const float* vec) const;

public:
    explicit SphericalGrid(int dimension);
    bool Add(const std::vector<float>& vec, const char* filePath);
    //std::vector<std::pair<WeightEntry, float>> FindNearestFull(const std::vector<float>& queryVec, int k);
    std::vector<std::pair<const WeightEntry*, float>> FindNearest(
        const std::vector<float>& queryVec, int k);
    std::vector<std::pair<const WeightEntry*, float>> FindNearestFull(const std::vector<float>& queryVec, int k);
    bool Delete(int id);
    bool Save();
    bool Load();
    size_t GetCount();
};