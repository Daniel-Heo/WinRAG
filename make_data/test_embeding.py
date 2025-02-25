import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# 사전 학습된 BERT 모델 로드
#MODEL_NAME = "jhgan/ko-sbert-nli"
MODEL_NAME = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# 입력 문장
sentence = "딥러닝을 통한 자연어 처리";
token_ids = tokenizer(sentence, return_tensors="pt")["input_ids"]
print("IDS:", token_ids)
print("토큰화 결과:", tokenizer.convert_ids_to_tokens(token_ids[0]))

# 사전 학습된 임베딩 가져오기
embedded_tokens = model.get_input_embeddings()(token_ids)
print("임베딩 크기:", embedded_tokens.shape)
print("임베딩 벡터:", embedded_tokens)
# 벡터 평균화 
word_embeddings = embedded_tokens.mean(dim=1)
print("평균 임베딩 크기:", word_embeddings.shape)
print("평균 임베딩 벡터:", word_embeddings)
