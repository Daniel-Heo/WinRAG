/*******************************************************************************
    파     일     명 : text_cluster_db.cpp
    프로그램명칭 :  유사도 DB 래퍼
    프로그램용도 : cluster_db에 대한 직접적인 접근을 하지 않고 이 클래스를 통해서 편리하게 사용. (유사도 DB 래퍼)
    참  고  사  항  :

    작    성    자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
    라 이 센 스  : MIT License
    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.23   Daniel Heo  최초 생성
*******************************************************************************/
#include "text_cluster_db.h"

TextClusterDB::TextClusterDB(int vectorDim, const char* relativePath)
    : clusterDB(vectorDim) {
    strcpy_s(savePath, relativePath);
}

bool TextClusterDB::InsertText(const std::vector<float>& vec, const std::string& str) {
    namespace fs = std::filesystem;
    std::error_code ec;
    int currentId;

	currentId = GetCurrentId();

    // 경로 생성
    fs::path dirPath(savePath);
    if (!fs::exists(dirPath)) {
        if (!fs::create_directories(dirPath, ec)) {
            std::cerr << "디렉토리 생성 실패: " << ec.message() << "\n";
            return false;
        }
    }

    // 파일 경로 설정: "savePath/currentId.txt"
    std::stringstream fileNameStream;
    fileNameStream << currentId << ".txt";
    fs::path relativeFilePath = dirPath / fileNameStream.str();

    // 파일로 저장
    std::ofstream outFile(relativeFilePath, std::ios::out | std::ios::trunc);
    if (!outFile) {
        std::cerr << "파일 저장 실패: " << relativeFilePath << "\n";
        return false;
    }

    outFile << str;
    outFile.close();

    // 내부 clusterDB에 상대 경로로 Add 호출
    if (!clusterDB.Add(vec, relativeFilePath.string().c_str())) {
        std::cerr << "clusterDB.Add 실패\n";
        return false;
    }

    return true;
}

std::string TextClusterDB::SearchText(const std::vector<float>& vec, int k) {
	std::string result;
	std::vector<std::pair<const WeightEntry*, float>> entries = clusterDB.FindNearestFull(vec, k);
	for (const auto& entry : entries) {
		std::ifstream inFile(entry.first->filePath);
		if (!inFile) {
			std::cerr << "파일 열기 실패: " << entry.first->filePath << "\n";
			continue;
		}
		std::string line;
		std::getline(inFile, line);
		result += line + "\n";
	}
	return result;
}

int test_text_cluster_db() {
    constexpr int VECTOR_DIM = 2048;  // 벡터 차원 설정
    constexpr int DATA_COUNT = 10000; // 데이터 개수

    TextClusterDB cdb(VECTOR_DIM, "db");

    // 임의의 데이터 생성
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vec(VECTOR_DIM);
    for (auto& val : vec) val = dist(gen);
    cdb.InsertText(vec, "안녕하세요");
    cdb.InsertText(vec, "반갑습니다");
    cdb.InsertText(vec, "좋은 아침입니다.");

    //cdb.Delete(1);
    printf("SDB Count: %d\n", cdb.GetCount());

    // 쿼리 생성 및 검색 수행
    std::vector<float> queryVec(VECTOR_DIM);
    for (auto& val : queryVec) val = dist(gen);

    // 검색
    std::string res = cdb.SearchText(queryVec, 1); // RunKMeansClustering 수행

	printf("검색 결과: %s\n", res.c_str());

    // 저장
    cdb.Save("cluster_db.bin");

    return 0;
}