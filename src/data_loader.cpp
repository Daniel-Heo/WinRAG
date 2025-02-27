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

using json = nlohmann::json;

/****************************************************************
* Function Name: parseCSVLine
* Description: CSV 한 줄을 파싱하여 필드별로 분리
* Parameters:
*   - line: CSV의 한 줄 문자열
* Return: 파싱된 문자열 리스트 (std::vector<std::string>)
****************************************************************/
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

/****************************************************************
* Function Name: extractKeyOrder
* Description: JSONL 파일에서 인자 순서를 추출
* Parameters:
*   - jsonString: JSONL 파일의 한 줄 문자열
* Return: 키 순서 리스트 (std::vector<std::string>)
****************************************************************/
std::vector<std::string> DataLoader::extractKeyOrder(const std::string& jsonString) {
    std::vector<std::string> keys;
    std::regex keyPattern(R"(\"([^\"]+)\":)"); // 키를 찾기 위한 정규 표현식
    std::sregex_iterator it(jsonString.begin(), jsonString.end(), keyPattern);
    std::sregex_iterator end;

    while (it != end) {
        keys.push_back(it->str(1)); // 첫 번째 캡처 그룹(키 이름)을 추가
        ++it;
    }
    return keys;
}

/****************************************************************
* Function Name: loadCSV
* Description: CSV 파일을 로드하여 데이터를 저장
* Parameters:
*   - filename: CSV 파일 경로 (wstring)
* Return: 파일 로드 성공 여부 (bool)
****************************************************************/
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

/****************************************************************
* Function Name: loadJSONL
* Description: JSONL 파일을 로드하여 데이터를 저장
*              - 첫 번째 행의 형식(배열 또는 객체) 기준으로 나머지 행을 파싱
*              - 첫 행이 배열이면 배열만, 객체이면 객체만 처리
*              - 필드 개수가 부족하면 "N/A"로 채움
****************************************************************/
bool DataLoader::loadJSONL(const std::wstring& filename) {
    HANDLE hFile = CreateFileW(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::wcerr << L"파일을 열 수 없습니다: " << filename << std::endl;
        return false;
    }

    DWORD fileSize = GetFileSize(hFile, NULL);
    if (fileSize == INVALID_FILE_SIZE) {
        CloseHandle(hFile);
        return false;
    }

    std::vector<char> buffer(fileSize + 1, '\0');
    DWORD bytesRead;
    if (!ReadFile(hFile, buffer.data(), fileSize, &bytesRead, NULL)) {
        CloseHandle(hFile);
        return false;
    }
    CloseHandle(hFile);

    try {
        std::stringstream ss(std::string(buffer.begin(), buffer.end()));
        std::string line;
        data.clear();
        columnCount = 0;
        bool isFirstRow = true;
        bool isObjectFormat = false;
        std::vector<std::string> columnNames;  // 객체 형식일 경우 키 목록 저장
        while (std::getline(ss, line)) {
            if (line.empty()) continue;

            try {
                nlohmann::json jsonData = nlohmann::json::parse(line);

                if (isFirstRow) {
                    isFirstRow = false;
                    if (jsonData.is_array()) {
                        // 첫 번째 행이 배열인 경우
                        columnCount = jsonData.size();
                        isObjectFormat = false;
                    }
                    else if (jsonData.is_object()) {
                        // 첫 번째 행이 객체인 경우
                        columnNames = extractKeyOrder(line);
                        columnCount = columnNames.size();
                        isObjectFormat = true;
                    }
                    else {
                        std::cerr << "JSONL 첫 번째 행이 배열 또는 객체가 아닙니다. 파싱 실패.\n";
                        return false;
                    }
                }

                if (isObjectFormat) {
                    // 객체 형식 처리
                    if (!jsonData.is_object()) {
                        std::cerr << "JSONL 형식 불일치: 첫 행이 객체인데 배열이 감지됨. 해당 줄 무시.\n";
                        continue;
                    }

                    std::vector<std::string> row(columnCount, "N/A");
                    for (size_t i = 0; i < columnCount; ++i) {
                        if (jsonData.contains(columnNames[i]) && jsonData[columnNames[i]].is_string()) {
                            row[i] = jsonData[columnNames[i]].get<std::string>();
                        }
                    }
                    data.emplace_back(row[0], (columnCount > 1 ? row[1] : "N/A"), (columnCount > 2 ? row[2] : "N/A"));
                }
                else {
                    // 배열 형식 처리
                    if (!jsonData.is_array()) {
                        std::cerr << "JSONL 형식 불일치: 첫 행이 배열인데 객체가 감지됨. 해당 줄 무시.\n";
                        continue;
                    }

                    std::vector<std::string> row(columnCount, "N/A");
                    for (size_t i = 0; i < std::min(columnCount, jsonData.size()); ++i) {
                        if (jsonData[i].is_string()) {
                            row[i] = jsonData[i].get<std::string>();
                        }
                    }
                    data.emplace_back(row[0], (columnCount > 1 ? row[1] : "N/A"), (columnCount > 2 ? row[2] : "N/A"));
                }
            }
            catch (const nlohmann::json::exception& e) {
                std::cerr << "JSONL 파싱 오류 (잘못된 줄 무시): " << e.what() << "\n";
                continue;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "JSONL 파일 읽기 오류: " << e.what() << std::endl;
        return false;
    }

    return true;
}

/****************************************************************
 * Function Name: loadTXT
 * Description: TXT 파일을 로드하여 데이터를 저장
 * Parameters:
 *  - filename: TXT 파일 경로 (wstring)
 * Return: 파일 로드 성공 여부 (bool)
 ****************************************************************/
bool DataLoader::loadTXT(const std::wstring& filename)  {
    HANDLE hFile = CreateFileW(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::wcerr << L"파일을 열 수 없습니다: " << filename << std::endl;
        return false;
    }

    DWORD fileSize = GetFileSize(hFile, NULL);
    if (fileSize == INVALID_FILE_SIZE) {
        CloseHandle(hFile);
        return false;
    }

    std::vector<char> buffer(fileSize + 1, '\0');
    DWORD bytesRead;
    if (!ReadFile(hFile, buffer.data(), fileSize, &bytesRead, NULL)) {
        CloseHandle(hFile);
        return false;
    }
    CloseHandle(hFile);

    std::string content(buffer.begin(), buffer.end());
    std::stringstream ss(content);
    std::string line;
    data.clear();
    columnCount = 1;
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        data.emplace_back(line, "N/A", "N/A");
    }

    return true;
}

/****************************************************************
* Function Name: Size
* Description: 저장된 데이터의 행과 열 개수를 반환
* Parameters: 없음
* Return: 데이터 행(row)과 열(column) 개수 (std::pair<size_t, size_t>)
****************************************************************/
std::pair<size_t, size_t> DataLoader::Size() const {
    return { data.size(), columnCount };
}

/****************************************************************
* Function Name: get
* Description: 특정 행의 데이터를 반환
* Parameters:
*   - row_num: 가져올 데이터의 행 번호 (size_t)
* Return: 해당 행의 데이터 (std::tuple<std::string, std::string, std::string>)
****************************************************************/
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
            std::cout << "  colum1: " << q << "\n";
            std::cout << "  colum2: " << a << "\n";
            std::cout << "  colum3: " << d << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
    }

    return 0;
}

// 테스트 코드 수정
int test_data_loader_formats() {
    DataLoader loader;

    // JSON 테스트
    //if (!loader.loadJSONL(L"train.jsonl")) {
    if (!loader.loadTXT(L"QA_total.csv")) {
        std::wcerr << L"JSON 파일을 불러오지 못했습니다." << std::endl;
    }

    auto [jsonRows, jsonCols] = loader.Size(); 
    std::cout << "JSON - Rows: " << jsonRows << ", Columns: " << jsonCols << std::endl;

    try {
        for (size_t i = 0; i < 10; ++i) {
            auto [q, a, d] = loader.get(i);
            std::cout << "Row " << i << ":\n";
            std::cout << "  colum1: " << q << "\n";
            std::cout << "  colum2: " << a << "\n";
            std::cout << "  colum3: " << d << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
    }

    return 0;
}