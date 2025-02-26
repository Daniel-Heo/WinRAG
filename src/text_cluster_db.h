/*******************************************************************************
    파     일     명 : text_cluster_db.h
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
#pragma once
#include "cluster_db.h"
#include <filesystem>
#include <fstream>
#include <sstream>

/****************************************************************
* Class Name: TextClusterDB
* Description: ClusterDB를 감싸는 래퍼 클래스.
*              텍스트를 파일로 저장하고, 유사도 기반 검색 기능 제공.
****************************************************************/
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

// Test function
int test_text_cluster_db();