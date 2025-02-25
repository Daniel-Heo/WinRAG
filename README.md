# WinRAG
 윈도우 개발자를 위한 Simple RAG 소스. 일반 PC에서 직접 RAG 기반 프로그램을 사용함으로써 보안성도 제공하고 사용자의 요구에 부합되는 다양한 프로그램을 개발하는데 도움이 될 것이라 생각해서 기본적인 토대를 만들어서 제공할 생각입니다. 참여하실 분은 환영합니다.

* 성능 데이터는 intel i5-12600K에서 테스트 되었습니다.

### 텍스트 토크나이저
Sentence Piece 토크나이저만 제공 : 한국어를 완벽하게 지원해줄 수 있어서. 
- IDS를 얻는 속도를 최적화에 중점 : Trie 알고리즘으로 token 검색 최적화와 메모리풀 사용
- 검색 성능 : 5만개의 vocab으로 10만건 검색 처리 시간("딮러닝은 AI 모델을 개선합니다.") -> 5초 정도

### 가중치 검색
AI 모델의 가중치 전체를 메모리에 1차원으로 정렬해서 올려두고 해당 ID에 대한 메모리 데이터를 제공
- 심플한 방식이라서 속도를 더 올릴만한 여지가 없음.

### 유사도 DB
데이터를 가중치 벡터를 연산하여 노멀라이징해서 유사도 DB에 넣은 후 특정 문장에 유사한 벡터들을 얻을 수 있는 DB입니다. 
```
1만건 입력 시간 : 0.827595s
클러스터링 재정렬 시간 : 0.297095s ( 모든 insert가 끝난 후 첫 검색 시에만 실행 )
검색 시간 : 0.156583s
풀스캔 검색시간 : 0.74529s
```

- 검색 방식 : 클러스터링(Clustering) 방식의 유사도 검색과 풀스캔 
- 유사도 계산 : SIMD로 처리 ( SSE2 / (AVX2+FMA3). 일반 PC에서 사용하는 것이라 심플한 형태를 취할 예정임. GPU는 적용하지 않을 예정. )
- 단점 : 완벽한 k - NN이 아닌 근사 결과 제공 -> 데이터가 1000개 미만일 경우에 풀스캔 함수 사용
- 저장/불러오기/추가/삭제 : 파일 이름을 직접 입력할 수 있음. ( 다양한 사용을 위해 )
- 기본 벡터 연산 함수 제공 : 
```cpp
void NormalizeVector(float* vec, size_t size); // 벡터 노멀라이즈
std::vector<float> MeanVector(std::vector<std::vector<float>>& matrix); // 평균 벡터 계산
float CosineSimilarity(const float* v1, const float* v2, size_t size); // 코사인 유사도 계산
```

## make_data 폴더
 - 모델->가중치 데이터 생성 : embedding_weight.npy
 - 토크나이저 파일 저장 : tokenizer.json
   
   save_weights.py : 가중치를 파일로 저장

   save_tokenizer.py : 토크나이저 파일 저장

## 사용 예제
```
    constexpr int VECTOR_DIM = 768;  // 벡터 차원 설정

    // 클래스 초기화
    WeightTokenizer weightTokenizer("embedding_weights.npy", "tokenizer.json");
    TextClusterDB cdb(VECTOR_DIM, "db"); // 현재 디렉토리 밑에 db라는 디렉토리에 텍스트 데이터가 id번호로 파일이 생성된다.

    // cdb.Load("cluster_db.bin"); // 저장된 DB 데이터를 사용하고 DataLoader를 사용하지 않는 경우
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
            averaged_weights = weightTokenizer.GetWeight(q.c_str()); // 가중치 가져오기
            cdb.InsertText(averaged_weights, a.c_str()); // DB에 저장
        }
    }
    catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
    }
    //cdb.Save("cluster_db.bin"); // 만들어진 DB 데이터를 저장한다.

    // 텍스트 입력 받아 검색 결과 출력
    std::string search_text;
    std::string res;
    while (1) {
        std::cout << "검색할 텍스트를 입력하세요: ";
        search_text = readUtf8FromConsole(); // readUtf8FromConsole()은 test_main.cpp에 구현
        search_text.erase(std::remove(search_text.begin(), search_text.end(), '\n'), search_text.end()); // \n 삭제
        averaged_weights = weightTokenizer.GetWeight(search_text); // 가중치 가져오기
        res = cdb.SearchText(averaged_weights, 1); // DB 검색
        printf("검색 결과: %s\n", res.c_str());
    }
```

