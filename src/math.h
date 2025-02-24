/*******************************************************************************
    파   일   명 : math.h
    프로그램명칭 :  vector 연산
    작   성   일 : 2025.2.22
    작   성   자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    프로그램용도 : 노멀라이징, 평균 계산, 코사인 유사도 계산

    참 고 사 항  :
    라 이 센 스  : MIT License

    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  최초 생성
    ----------------------------------------------------------------------------

*******************************************************************************/
#pragma once
#include <iostream>
#include <vector>
#include <numeric>  // std::inner_product()
#include <cmath>  // sqrt()

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

// 각 열의 평균을 구하는 함수
std::vector<float> MeanVector(std::vector<std::vector<float>>& matrix);
// 두개의 벡터의 cosine similarity를 구하는 함수
float CosineSimilarity(const float* v1, const float* v2, size_t size);
// 벡터의 크기를 1로 만드는 함수
void NormalizeVector(float* vec, size_t size);

// Test function
int test_math();