# WinRAG
윈도우 개발자를 위한 Simple RAG 소스

- Tokenizer : 문장 입력시 IDS 리턴
- Embedding Weights : IDS에 해당하는 가중치 결과들을 리턴.

## Tokenizer
 - Wordpiece tokenizer만 지원 ( JSON Tokenizer 파일 )
 - 적용 알고리즘 : Trie, 메모리풀
 - 검색 성능 : intel i5-12600K에서 5만개의 vocab으로 10만건 검색 처리 시간("딮러닝은 AI 모델을 개선합니다.") -> 5초 정도
 - UTF8 지원

## make_data 폴더
 - 모델->가중치 데이터 생성 : embedding_weight.npy
   
   weights_save.py : 가중치를 파일로 저장

## 지원 함수 ( CPP )
 - test_token.cpp : 텍스트로 토큰 IDS와 토크나이징된 분리된 문자열을 가져오는 예제 소스
 - test_weights.cpp : IDS로 가중치를 차원 개수만큼 가져오는 예제소스

## TODO List
 - Faiss 적용 : https://github.com/facebookresearch/faiss
 - 평가 라이브러리 적용
