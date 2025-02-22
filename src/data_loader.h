/*******************************************************************************
    ��   ��   �� : data_loader.h
    ���α׷���Ī : ������ ���� �δ� (CSV)
    ��   ��   �� : 2025.2.22
    ��   ��   �� : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    ���α׷��뵵 : CSV ������ �ε��Ͽ� �����͸� �о���� Ŭ����
    �� �� �� ��  :
    �� �� �� ��  : MIT License

    ----------------------------------------------------------------------------
    ��������    ������      ��������
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  ���� ����
    ----------------------------------------------------------------------------

*******************************************************************************/
#pragma once
#include <windows.h>
#include <vector>
#include <string>
#include <tuple>
#include <iostream>

class DataLoader {
private:
    std::vector<std::tuple<std::string, std::string, std::string>> data;
    size_t columnCount = 0;

    // ���ڿ��� �Ľ��ϴ� �Լ�
    std::vector<std::string> parseCSVLine(const std::string& line);

public:
    // CSV ������ �ε��ϴ� �Լ�
    bool loadCSV(const std::wstring& filename);

    // ����� �������� ũ�� ��ȯ
    std::pair<size_t, size_t> Size() const;

    // Ư�� row �����͸� ��ȯ
    std::tuple<std::string, std::string, std::string> get(size_t row_num) const;
};

int test_data_loader();