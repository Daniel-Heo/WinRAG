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

#define INDEX_FILENAME "sdb.index"
#define LINK_FILENAME "sdb.data"

void NormalizeVector(float* vec, size_t size); // 벡터 노멀라이즈
std::vector<float> MeanVector(std::vector<std::vector<float>>& matrix); // 평균 벡터 계산
float CosineSimilarity(const float* v1, const float* v2, size_t size); // 코사인 유사도 계산

int test_similarity_db();
int test_mean();