#pragma once
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

#define SIMD_TYPE 1 // 0: SSE2 사용,  1: AVX2+FMA3 사용

#if SIMD_TYPE == 1
#include <immintrin.h>  // AVX, AVX2, FMA3, SSE 관련 헤더
#else
#include <emmintrin.h> // SSE2
#endif

#define INDEX_FILENAME "sdb.index"
#define LINK_FILENAME "sdb.data"

int test_similarity_db();
int test_mean();