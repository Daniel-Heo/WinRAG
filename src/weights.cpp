// weights.cpp
#include "weights.h"

// float16 -> float32 변환 함수
float float16_to_float32(uint16_t h) {
    // IEEE 754 half-precision layout
    uint16_t sign = (h & 0x8000) >> 15;
    uint16_t exponent = (h & 0x7C00) >> 10;
    uint16_t mantissa = h & 0x03FF;

    float value;
    if (exponent == 0) {
        // 서브노멀 (Subnormal)
        value = std::ldexp(mantissa, -24);
    }
    else if (exponent == 0x1F) {
        // NaN or Infinity
        value = mantissa ? NAN : INFINITY;
    }
    else {
        // 일반적인 float 값
        value = std::ldexp(mantissa + 1024, exponent - 25);
    }

    return sign ? -value : value;
}

// NumPy .npy 파일 float16 → float32 로드
std::vector<float> load_npy_float16_to_float32(const std::string& filename, size_t& rows, size_t& cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("파일을 열 수 없습니다: " + filename);
    }

    // 1. NPY 헤더 확인
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("잘못된 NPY 파일 형식입니다.");
    }

    // 2. 버전 및 헤더 길이 읽기
    unsigned char major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    unsigned short header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    // 3. 헤더 읽기
    std::string header(header_len, ' ');
    file.read(&header[0], header_len);

    // 4. dtype(`float16`: '<f2') 확인
    std::regex descr_pattern(R"('descr':\s*'([^']+)')");
    std::smatch descr_match;
    if (std::regex_search(header, descr_match, descr_pattern)) {
        std::string dtype = descr_match[1].str();
        if (dtype != "<f2" && dtype != "|f2") {
            throw std::runtime_error("지원되지 않는 데이터 타입입니다: " + dtype);
        }
    }
    else {
        throw std::runtime_error("데이터 타입(`descr`) 파싱 실패.");
    }

    // 5. 배열 크기(`shape`) 확인
    std::regex shape_pattern(R"('shape':\s*\((\d+),\s*(\d+)\))");
    std::smatch shape_match;
    if (std::regex_search(header, shape_match, shape_pattern)) {
        rows = std::stoul(shape_match[1].str());
        cols = std::stoul(shape_match[2].str());
    }
    else {
        throw std::runtime_error("배열 크기(`shape`) 파싱 실패.");
    }

    std::cout << "배열 크기: " << rows << " x " << cols << std::endl;

    // 6. float16 데이터 즉시 float32로 변환하며 읽기
    size_t total_elements = rows * cols;
    std::vector<float> float32_data(total_elements);

    for (size_t i = 0; i < total_elements; ++i) {
        uint16_t h;
        file.read(reinterpret_cast<char*>(&h), sizeof(uint16_t));
        float32_data[i] = float16_to_float32(h);
    }

    file.close();
    return float32_data;
}
