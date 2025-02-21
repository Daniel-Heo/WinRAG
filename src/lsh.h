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
#include <emmintrin.h> // SSE2
#include <memory>      // aligned_alloc
#include <functional>

#define INDEX_FILENAME "vector_db.index"
#define LINK_FILENAME "vector_db.data"

int test_lsh();