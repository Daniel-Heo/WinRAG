/*******************************************************************************
    파     일     명 : weight_loader.cpp
    프로그램명칭 :  가중치 데이터 로드 클래스
    프로그램용도 : 가중치 데이터를 로드하여 저장/검색하는 클래스
    참  고  사  항  :

    작    성    자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    라 이 센 스  : MIT License
    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  최초 생성
*******************************************************************************/
#include "weight_loader.h"

// float16 → float32 변환 함수 (private)
float WeightLoader::float16_to_float32(uint16_t h) {
    uint16_t sign = (h & 0x8000) >> 15;
    uint16_t exponent = (h & 0x7C00) >> 10;
    uint16_t mantissa = h & 0x03FF;

    float value;
    if (exponent == 0) {
        value = std::ldexp(mantissa, -24);  // 서브노멀 (Subnormal)
    }
    else if (exponent == 0x1F) {
        value = mantissa ? NAN : INFINITY;  // NaN 또는 Infinity 처리
    }
    else {
        value = std::ldexp(mantissa + 1024, exponent - 25);
    }

    return sign ? -value : value;
}

// 생성자: 가중치 파일 로드
WeightLoader::WeightLoader(const std::string& filename) {
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

    //std::cout << "배열 크기: " << rows << " x " << cols << std::endl;

    // 6. float16 데이터 즉시 float32로 변환하며 읽기
    size_t total_elements = rows * cols;
    weights.resize(total_elements);

    for (size_t i = 0; i < total_elements; ++i) {
        uint16_t h;
        file.read(reinterpret_cast<char*>(&h), sizeof(uint16_t));
        weights[i] = float16_to_float32(h);
    }

    file.close();
}

// 특정 위치(row, col)의 가중치 반환
float WeightLoader::get(int row_idx, int col_idx) const {
    if (row_idx < 0 || row_idx >= rows || col_idx < 0 || col_idx >= cols) {
        throw std::out_of_range("잘못된 인덱스입니다.");
    }
    return weights[row_idx * cols + col_idx];
}

std::vector<float> WeightLoader::get(int row_idx) const {
    if (row_idx < 0 || row_idx >= rows ) {
        throw std::out_of_range("잘못된 인덱스입니다.");
    }
    
	std::vector<float> result;
	for (int i = 0; i < cols; i++) {
		result.push_back(weights[row_idx * cols + i]);
	}
	return result;
}


// 행 개수 반환
int WeightLoader::get_rows() const {
    return rows;
}

// 열 개수 반환
int WeightLoader::get_cols() const {
    return cols;
}

// 가중치 데이터 개수 반환
int WeightLoader::get_size() const {
	return static_cast<int>(weights.size());
}


int test_weights() {
    try {
        WeightLoader loader("embedding_weights.npy");

        std::cout << "총 행 개수: " << loader.get_rows() << std::endl;
        std::cout << "총 열 개수: " << loader.get_cols() << std::endl;
        std::cout << "총 가중치 개수: " << loader.get_size() << std::endl;

        // 특정 가중치 값 가져오기
        int row = 0, col = 2;
        float weight = loader.get(row, col);
        std::cout << "Weight[" << row << "][" << col << "]: " << weight << std::endl;
		std::vector<float> row_weights = loader.get(row);
		std::cout << "Weight Data[";
        for (const auto& w : row_weights) {
            std::cout << w << " ";
        }
		std::cout << "]" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
    }

    return 0;
}
