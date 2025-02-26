/*******************************************************************************
    파     일     명 : test_main.h
    프로그램명칭 :  테스트 메인
    프로그램용도 : 단위테스트 및 복합 테스트
    참  고  사  항  :

    작    성    자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    라 이 센 스  : MIT License
    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  최초 생성
*******************************************************************************/
//#include "tokenizer.h"
//#include "weight_loader.h"
//#include "cluster_db.h"
//#include "bm25.h"
#include "data_loader.h"
#include "text_cluster_db.h"
#include "weight_tokenizer.h"

// 16진수 문자열 출력
void printHexString(const std::string& str) {
    char temp[1024] = { 0 }; // 문자열을 저장할 배열
    size_t length = str.copy(temp, sizeof(temp) - 1); // 문자열 복사 (초과 방지)
    temp[length] = '\0'; // 널 종료 문자 추가

    std::cout << "HEX: ";
    for (size_t i = 0; i < length; ++i) {
        unsigned char byte = static_cast<unsigned char>(temp[i]); // 부호 없는 바이트로 변환
        char highNibble = (byte >> 4) & 0x0F; // 상위 4비트
        char lowNibble = byte & 0x0F;        // 하위 4비트

        // 16진수 문자로 변환하여 출력
        std::cout << std::hex << std::uppercase
            << static_cast<char>(highNibble < 10 ? '0' + highNibble : 'A' + highNibble - 10)
            << static_cast<char>(lowNibble < 10 ? '0' + lowNibble : 'A' + lowNibble - 10);
    }
    std::cout << std::endl;
}

// 콘솔  입력을 UTF-8로 읽어오는 함수
std::string readUtf8FromConsole() {
    // 1. 콘솔 입력을 UTF-16으로 받기 위한 설정
    constexpr DWORD BUFFER_SIZE = 256;
    wchar_t wideBuffer[BUFFER_SIZE] = { 0 }; // UTF-16 버퍼
    DWORD charsRead = 0;

    HANDLE hInput = GetStdHandle(STD_INPUT_HANDLE);
    if (hInput == INVALID_HANDLE_VALUE) {
        std::cerr << "콘솔 입력 핸들을 가져오는 데 실패했습니다.\n";
        return "";
    }

    std::wcout << L"입력: ";
    if (!ReadConsoleW(hInput, wideBuffer, BUFFER_SIZE - 1, &charsRead, nullptr)) {
        std::cerr << "콘솔 입력을 읽는 데 실패했습니다.\n";
        return "";
    }

    // 개행 문자 제거
    if (charsRead > 0 && wideBuffer[charsRead - 1] == L'\n') {
        wideBuffer[charsRead - 1] = L'\0';
    }

    // 2. UTF-16 → UTF-8 변환
    int utf8Size = WideCharToMultiByte(CP_UTF8, 0, wideBuffer, -1, nullptr, 0, nullptr, nullptr);
    if (utf8Size <= 0) {
        std::cerr << "UTF-8 변환 크기를 계산하는 데 실패했습니다.\n";
        return "";
    }

    std::vector<char> utf8Buffer(utf8Size);
    WideCharToMultiByte(CP_UTF8, 0, wideBuffer, -1, utf8Buffer.data(), utf8Size, nullptr, nullptr);

    return std::string(utf8Buffer.data()); // 변환된 UTF-8 문자열 반환
}

// WinRAG 예제
int test_main() {
    constexpr int VECTOR_DIM = 768;  // 벡터 차원 설정

    // 클래스 초기화
    WeightTokenizer weightTokenizer("embedding_weights.npy", "tokenizer.json");
    TextClusterDB cdb(VECTOR_DIM, "db");

	// cdb.Load("cluster_db.bin");
    DataLoader loader;

	// CSV 파일 로드
    if (!loader.loadCSV(L"QA_total.csv")) {
        std::wcerr << L"CSV 파일을 불러오지 못했습니다." << std::endl;
        return 1;
    }
    auto [rows, cols] = loader.Size();
    std::cout << "Rows: " << rows << ", Columns: " << cols << std::endl;

	// DB에 데이터 추가
    std::vector<float> averaged_weights;
    try {
        for (size_t i = 0; i < rows; ++i) {
            auto [q, a, d] = loader.get(i);
            averaged_weights = weightTokenizer.GetWeight(q.c_str());
            cdb.InsertText(averaged_weights, a.c_str());
        }
    }
    catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
    }
	//cdb.Save("cluster_db.bin");

	// 검색할 텍스트 입력 받기
	std::string search_text;
    std::string res;
    while (1) {
        std::cout << "검색할 텍스트를 입력하세요: ";
        search_text = readUtf8FromConsole();
        search_text.erase(std::remove(search_text.begin(), search_text.end(), '\n'), search_text.end());
        averaged_weights = weightTokenizer.GetWeight(search_text);
        res = cdb.SearchText(averaged_weights, 1);
        printf("검색 결과: %s\n", res.c_str());
    }

    return 0;
}

int main() {
    // 콘솔의 입력과 출력 코드 페이지를 UTF-8로 변경
    SetConsoleCP(CP_UTF8);         // 입력 (키보드)
    SetConsoleOutputCP(CP_UTF8);   // 출력 (콘솔)
    
    //test_tokenizer();
	//test_weights();
    //test_cluster_db();
    //test_cluster_db_accuracy();
	//test_mean();
    //test_bm25();
	//test_data_loader();
    //test_text_cluster_db();
    //test_weight_tokenizer();
    test_main();
    
    return 0;
}


