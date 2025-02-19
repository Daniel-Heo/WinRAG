from transformers import AutoModel
import numpy as np

# 사용할 모델 (DeepSeek-R1 14B)
MODEL_NAME = "skt/kogpt2-base-v2"

# 모델 불러오기
model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)

# 임베딩 레이어 가중치 추출 및 저장
embedding_weights = model.get_input_embeddings().weight.detach().cpu().numpy()

# 5120 차원을 2개씩 평균내어 2560 차원으로 축소 : 1.5G(5120) -> 778M(2560) -> 389M(1280) -> 194M(640) -> 97M(320) -> 48M(160)
# (152064, 5120) -> (152064, 2560)
#compressed_weights = embedding_weights.reshape(152064, 2560, 2).mean(axis=2) # deepseek r1 14B
compressed_weights = embedding_weights.reshape(51200, 192, 4).mean(axis=2) # kogpt2-base-v2

# float16으로 변환
embedding_weights_fp16 = compressed_weights.astype(np.float16)

print("임베딩 가중치 변환 완료:", embedding_weights_fp16.shape)

np.save("embedding_weights.npy", embedding_weights_fp16)  # Numpy 배열로 저장