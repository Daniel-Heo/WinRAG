/*******************************************************************************
    파     일     명 : data_loader.cpp
    프로그램명칭 :  데이터 파일 로더 (CSV)
    프로그램용도 : CSV 파일을 로드하여 데이터를 읽어오는 클래스
    참  고  사  항  :

    작    성    자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    라 이 센 스  : MIT License
    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  최초 생성
*******************************************************************************/
#include "data_loader.h"

// CSV 한 줄을 파싱하는 함수
std::vector<std::string> DataLoader::parseCSVLine(const std::string& line) {
    std::vector<std::string> row;
    std::string field;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];

        if (c == '"') {
            inQuotes = !inQuotes;
        }
        else if (c == ',' && !inQuotes) {
            row.push_back(field);
            field.clear();
        }
        else {
            field += c;
        }
    }
    row.push_back(field);  // 마지막 필드 추가

    return row;
}

// CSV 파일을 로드하는 함수
bool DataLoader::loadCSV(const std::wstring& filename) {
    // 파일 핸들 생성
    HANDLE hFile = CreateFileW(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::wcerr << L"파일을 열 수 없습니다: " << filename << std::endl;
        return false;
    }

    // 파일 크기 확인
    DWORD fileSize = GetFileSize(hFile, NULL);
    if (fileSize == INVALID_FILE_SIZE) {
        CloseHandle(hFile);
        return false;
    }

    // 버퍼 할당 및 파일 읽기
    std::vector<char> buffer(fileSize + 1, '\0');
    DWORD bytesRead;
    if (!ReadFile(hFile, buffer.data(), fileSize, &bytesRead, NULL)) {
        CloseHandle(hFile);
        return false;
    }
    CloseHandle(hFile);

    // 파일 내용을 문자열로 변환
    std::string content(buffer.begin(), buffer.end());

    // 행 단위로 나누기
    size_t start = 0, end = 0;
    bool isHeader = true;
    while ((end = content.find('\n', start)) != std::string::npos) {
        std::string line = content.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();  // CRLF 처리
        }
        start = end + 1;

        auto row = parseCSVLine(line);
        if (isHeader) {
            columnCount = row.size(); // 첫 번째 행(헤더)의 컬럼 개수를 저장
            isHeader = false;
            continue;
        }

        if (row.size() == 3) {
            data.emplace_back(row[0], row[1], row[2]);
        }
    }

    return true;
}

// 저장된 데이터의 크기 반환
std::pair<size_t, size_t> DataLoader::Size() const {
    return { data.size(), columnCount };
}

// 특정 row 데이터를 반환
std::tuple<std::string, std::string, std::string> DataLoader::get(size_t row_num) const {
    if (row_num >= data.size()) {
        throw std::out_of_range("인덱스 범위를 초과했습니다.");
    }
    return data[row_num];
}

// 테스트 코드
int test_data_loader() {
    DataLoader loader;
    if (!loader.loadCSV(L"QA_total.csv")) {
        std::wcerr << L"CSV 파일을 불러오지 못했습니다." << std::endl;
        return 1;
    }

    auto [rows, cols] = loader.Size();
    std::cout << "Rows: " << rows << ", Columns: " << cols << std::endl;

    try {
        for (size_t i = 0; i < rows; ++i) {
            auto [q, a, d] = loader.get(i);
            std::cout << "Row " << i << ":\n";
            std::cout << "  Question: " << q << "\n";
            std::cout << "  Answer: " << a << "\n";
            std::cout << "  Documents: " << d << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
    }

    return 0;
}
