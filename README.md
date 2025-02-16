# WinRAG
윈도우 개발자를 위한 Simple RAG 소스

- Tokenizer : 문장 입력시 IDS 리턴
- Embedding Weights : IDS에 해당하는 가중치 결과들을 리턴.

## make_data 폴더
 - 모델->가중치 데이터 생성 : embedding_weight.npy
 - Tokenizer 파일로 토크나이저 JSON 파일 생성

   embedding_save.py : 가중치를 파일로 저장
   
   token_save.py : 토크나이저를 파일로 저장

## 지원 함수
 - token_test : 텍스트로 토큰 IDS와 토크나이징된 분리된 문자열을 가져오는 예제 함수.
 - weights_test : IDS로 가중치를 차원 개수만큼 가져오는 예제함수.
