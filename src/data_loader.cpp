/*******************************************************************************
    ��     ��     �� : data_loader.cpp
    ���α׷���Ī :  ������ ���� �δ� (CSV)
    ���α׷��뵵 : CSV ������ �ε��Ͽ� �����͸� �о���� Ŭ����
    ��  ��  ��  ��  :

    ��    ��    �� : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    �� �� �� ��  : MIT License
    ----------------------------------------------------------------------------
    ��������    ������      ��������
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  ���� ����
*******************************************************************************/
#include "data_loader.h"

// CSV �� ���� �Ľ��ϴ� �Լ�
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
    row.push_back(field);  // ������ �ʵ� �߰�

    return row;
}

// CSV ������ �ε��ϴ� �Լ�
bool DataLoader::loadCSV(const std::wstring& filename) {
    // ���� �ڵ� ����
    HANDLE hFile = CreateFileW(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::wcerr << L"������ �� �� �����ϴ�: " << filename << std::endl;
        return false;
    }

    // ���� ũ�� Ȯ��
    DWORD fileSize = GetFileSize(hFile, NULL);
    if (fileSize == INVALID_FILE_SIZE) {
        CloseHandle(hFile);
        return false;
    }

    // ���� �Ҵ� �� ���� �б�
    std::vector<char> buffer(fileSize + 1, '\0');
    DWORD bytesRead;
    if (!ReadFile(hFile, buffer.data(), fileSize, &bytesRead, NULL)) {
        CloseHandle(hFile);
        return false;
    }
    CloseHandle(hFile);

    // ���� ������ ���ڿ��� ��ȯ
    std::string content(buffer.begin(), buffer.end());

    // �� ������ ������
    size_t start = 0, end = 0;
    bool isHeader = true;
    while ((end = content.find('\n', start)) != std::string::npos) {
        std::string line = content.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();  // CRLF ó��
        }
        start = end + 1;

        auto row = parseCSVLine(line);
        if (isHeader) {
            columnCount = row.size(); // ù ��° ��(���)�� �÷� ������ ����
            isHeader = false;
            continue;
        }

        if (row.size() == 3) {
            data.emplace_back(row[0], row[1], row[2]);
        }
    }

    return true;
}

// ����� �������� ũ�� ��ȯ
std::pair<size_t, size_t> DataLoader::Size() const {
    return { data.size(), columnCount };
}

// Ư�� row �����͸� ��ȯ
std::tuple<std::string, std::string, std::string> DataLoader::get(size_t row_num) const {
    if (row_num >= data.size()) {
        throw std::out_of_range("�ε��� ������ �ʰ��߽��ϴ�.");
    }
    return data[row_num];
}

// �׽�Ʈ �ڵ�
int test_data_loader() {
    DataLoader loader;
    if (!loader.loadCSV(L"QA_total.csv")) {
        std::wcerr << L"CSV ������ �ҷ����� ���߽��ϴ�." << std::endl;
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
        std::cerr << "���� �߻�: " << e.what() << std::endl;
    }

    return 0;
}
