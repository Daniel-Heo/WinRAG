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

#define SIMD_TYPE 1 // 0: SSE2 ���,  1: AVX2+FMA3 ���

#if SIMD_TYPE == 1
#include <immintrin.h>  // AVX, AVX2, FMA3, SSE ���� ���
#else
#include <emmintrin.h> // SSE2
#endif

#define INDEX_FILENAME "vector_db.index"
#define LINK_FILENAME "vector_db.data"

int test_lsh();