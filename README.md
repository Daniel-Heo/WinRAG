# WinRAG
윈도우 개발자를 위한 Simple RAG 소스. 일반 PC에서 직접 RAG 기반 프로그램을 사용함으로써 보안성도 제공하고 사용자의 요구에 부합되는 다양한 프로그램을 개발하는데 도움이 될 것이라 생각해서 기본적인 토대를 만들어서 제공할 생각입니다. 참여하실 분은 환영합니다.

### 텍스트 토크나이저
Sentence Piece 토크나이저만 제공 : 한국어를 완벽하게 지원해줄 수 있어서. 
- IDS를 얻는 속도를 최적화에 중점 : Trie 알고리즘으로 token 검색 최적화와 메모리풀 사용
- 검색 성능 : intel i5-12600K에서 5만개의 vocab으로 10만건 검색 처리 시간("딮러닝은 AI 모델을 개선합니다.") -> 5초 정도

### 가중치 검색
AI 모델의 가중치 전체를 메모리에 1차원으로 정렬해서 올려두고 해당 ID에 대한 메모리 데이터를 제공
- 심플한 방식이라서 속도를 더 올릴만한 여지가 없음.

### 유사도 DB
데이터를 가중치 벡터를 연산하여 노멀라이징해서 유사도 DB에 넣은 후 특정 문장에 유사한 벡터들을 얻을 수 있는 DB입니다. 
```
1만건 입력 시간 : 0.827595s
검색 시간 : 0.453678s
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

## 테스트
 - test_main.cpp : 공통으로 테스트 진행중

## TODO List
 - Similarity DB에 입력하는 방식을 datasets로 규칙성을 가지고 해당 pdf나 txt파일을 잘라내서 자동적으로 넣을 수 있는 라이브러리 개발
