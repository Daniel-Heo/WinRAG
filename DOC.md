📌 WinRAG 프로젝트 문서

작성자: Daniel Heo

라이선스: MIT License

최종 수정일: 2025-02-23

1️⃣ 프로젝트 개요

WinRAG 프로젝트는 클러스터링을 기반으로 한 유사도 검색 시스템입니다.

다음과 같은 기능을 제공합니다.

- 가중치 벡터 로드 및 변환 (Numpy .npy 파일 로드)
- 텍스트 토크나이징 (SentencePiece 기반)
- 텍스트와 가중치를 연계하여 저장
- K-Nearest Neighbor 기반 유사도 검색
- 코사인 유사도, 평균 벡터 연산

2️⃣ 파일 설명
파일명	설명
cluster_db.h/cpp	클러스터링 기반 데이터베이스 구현
data_loader.h/cpp	CSV 데이터를 로드하고 저장
text_cluster_db.h/cpp	클러스터링 DB의 고수준 API 제공
tokenizer.h/cpp	문장을 토큰화하여 벡터로 변환
weight_loader.h/cpp	가중치 데이터를 로드하고 변환
weight_tokenizer.h/cpp	토큰화된 텍스트를 가중치 벡터로 변환
math.h/cpp	벡터 연산 (정규화, 평균, 코사인 유사도)
test_main.cpp	프로젝트 전체 테스트 코드
3️⃣ 클래스 및 주요 함수 설명
📌 클러스터링 데이터베이스 (cluster_db.h)
Class: ClusterDB
✔ Add() - 새 데이터를 추가
✔ FindNearestCluster() - 클러스터 내에서 가장 유사한 데이터 검색
✔ FindNearestFull() - 전체 데이터에서 가장 유사한 데이터 검색

📌 CSV 데이터 로더 (data_loader.h)
Class: DataLoader
✔ loadCSV() - CSV 파일을 로드
✔ get() - 특정 행의 데이터 반환
✔ Size() - 데이터 크기 반환

📌 텍스트 클러스터링 래퍼 (text_cluster_db.h)
Class: TextClusterDB
✔ InsertText() - 텍스트와 벡터를 함께 저장
✔ SearchText() - 유사한 텍스트 검색

📌 토크나이저 (tokenizer.h)
Class: Tokenizer
✔ loadTokenizer() - JSON 형식의 토크나이저 로드
✔ tokenize() - 문장을 토큰으로 변환

📌 가중치 로더 (weight_loader.h)
Class: WeightLoader
✔ get() - 특정 행 또는 셀의 가중치 반환

📌 가중치 토크나이저 (weight_tokenizer.h)
Class: WeightTokenizer
✔ GetWeight() - 문장을 가중치 벡터로 변환

📌 벡터 연산 (math.h)
✔ NormalizeVector() - 벡터 정규화
✔ MeanVector() - 벡터 평균 계산
✔ CosineSimilarity() - 두 벡터 간 코사인 유사도 계산

4️⃣ 사용 방법
✅ 1. 프로젝트 빌드
bash
복사
편집
g++ -o main.exe test_main.cpp cluster_db.cpp data_loader.cpp \
    text_cluster_db.cpp tokenizer.cpp weight_loader.cpp \
    weight_tokenizer.cpp math.cpp -std=c++20 -mavx2
✅ 2. 실행
bash
복사
편집
./main.exe
5️⃣ 테스트 코드 실행 방법
📌 개별 테스트 실행
cpp
복사
편집
test_tokenizer();
test_weights();
test_data_loader();
test_text_cluster_db();
test_weight_tokenizer();
test_math();
📌 전체 테스트 실행 (test_main.cpp)
cpp
복사
편집
int main() {
    test_main();
    return 0;
}
