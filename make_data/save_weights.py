from transformers import AutoModel
import numpy as np

# 사용할 모델 (DeepSeek-R1 14B)
MODEL_NAME = "skt/kogpt2-base-v2"
#MODEL_NAME = "jhgan/ko-sbert-nli"

# 모델 불러오기
model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)

# 임베딩 레이어 가중치 추출 및 저장
embedding_weights = model.get_input_embeddings().weight.detach().cpu().numpy()

# 차원 평균으로 축소하기
#compressed_weights = embedding_weights.reshape(51200, 192, 4).mean(axis=2) # kogpt2-base-v2

# 그대로 사용 : 51200 vocab에 768차원 : 51200*768*2 = 78MB
compressed_weights = embedding_weights

# float16으로 변환
embedding_weights_fp16 = compressed_weights.astype(np.float16)

print("임베딩 가중치 변환 완료:", embedding_weights_fp16.shape)

np.save("embedding_weights.npy", embedding_weights_fp16)  # Numpy 배열로 저장