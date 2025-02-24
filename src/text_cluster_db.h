#pragma once
#include "cluster_db.h"
#include <filesystem>
#include <fstream>
#include <sstream>

class TextClusterDB {
private:
    ClusterDB clusterDB; // 내부 DB 객체 소유
	char savePath[MAX_FILE_PATH]; // 저장 경로 ( 상대 경로 )

public:
    TextClusterDB(int vectorDim, const char* relativePath);

    bool InsertText(const std::vector<float>& vec, const std::string& str);
	std::string SearchText(const std::vector<float>& vec, int k);


    bool Save(const char* filename) { return clusterDB.Save(filename); }
    bool Load(const char* filename) { return clusterDB.Load(filename); }
	int GetCurrentId() { return clusterDB.GetCurrentId(); }
	int GetCount() { return clusterDB.GetCount(); }

    // ClusterDB의 추가 기능 필요하면 래핑해서 노출
};

int test_text_cluster_db();