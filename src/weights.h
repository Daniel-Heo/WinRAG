// weights.h
#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <regex>
#include <cmath>

// 가중치 데이터 로드 클래스
class WeightLoader {
private:
    std::vector<float> weights;  // 가중치 저장 (float32 데이터)
    int rows = 0, cols = 0;      // 행, 열 크기

    // float16 → float32 변환 함수
    float float16_to_float32(uint16_t h);

public:
    // 생성자: 파일을 로드하여 가중치를 저장
    explicit WeightLoader(const std::string& filename);

    // 특정 위치(row, col)의 가중치 반환
    float get(int row_idx, int col_idx) const;
    std::vector<float> get(int row_idx) const;

    // 행, 열 크기 반환
    int get_rows() const;
    int get_cols() const;
	int get_size() const;
};

// Test function
int test_weights();
