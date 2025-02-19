#pragma once
#include <iostream>
#include <vector>
#include <numeric>  // std::inner_product()
#include <cmath>  // sqrt()

// 각 열의 평균을 구하는 함수
std::vector<float> MeanOfMatrix(std::vector<std::vector<float>> matrix);
// 두개의 벡터의 cosine similarity를 구하는 함수
float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
// 벡터의 크기를 1로 만드는 함수
std::vector<float> Normalize(std::vector<float> a);

// Test function
int test_math();