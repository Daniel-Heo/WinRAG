/*******************************************************************************
    파     일     명 : data_loader.h
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
#pragma once
#include <windows.h>
#include <vector>
#include <string>
#include <tuple>
#include <iostream>

/****************************************************************
* Class Name: DataLoader
* Description: CSV 파일을 로드하여 데이터를 저장하고 제공하는 클래스
****************************************************************/
class DataLoader {
private:
    std::vector<std::tuple<std::string, std::string, std::string>> data;
    size_t columnCount = 0;

    // 문자열을 파싱하는 함수
    std::vector<std::string> parseCSVLine(const std::string& line);

public:
    // CSV 파일을 로드하는 함수
    bool loadCSV(const std::wstring& filename);

    // 저장된 데이터의 크기 반환
    std::pair<size_t, size_t> Size() const;

    // 특정 row 데이터를 반환
    std::tuple<std::string, std::string, std::string> get(size_t row_num) const;
};

// Test function
int test_data_loader();