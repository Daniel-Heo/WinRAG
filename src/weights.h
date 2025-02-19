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

// ����ġ ������ �ε� Ŭ����
class WeightLoader {
private:
    std::vector<float> weights;  // ����ġ ���� (float32 ������)
    int rows = 0, cols = 0;      // ��, �� ũ��

    // float16 �� float32 ��ȯ �Լ�
    float float16_to_float32(uint16_t h);

public:
    // ������: ������ �ε��Ͽ� ����ġ�� ����
    explicit WeightLoader(const std::string& filename);

    // Ư�� ��ġ(row, col)�� ����ġ ��ȯ
    float get(int row_idx, int col_idx) const;
    std::vector<float> get(int row_idx) const;

    // ��, �� ũ�� ��ȯ
    int get_rows() const;
    int get_cols() const;
	int get_size() const;
};

// Test function
int test_weights();
