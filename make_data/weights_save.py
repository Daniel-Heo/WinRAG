from transformers import AutoModel
import numpy as np

# 사용할 모델 (DeepSeek-R1 14B)
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# 모델 불러오기
model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)

# 임베딩 레이어 가중치 추출 및 저장
embedding_weights = model.get_input_embeddings().weight.detach().cpu().numpy()

# 8192 차원을 4개씩 평균내어 2048 차원으로 축소 : 2.5G(8192) -> 625M(2048) -> 312M(1024) -> 156M(512)
# (152064, 8192) -> (152064, 2048)
compressed_weights = embedding_weights.reshape(152064, 2048, 4).mean(axis=2)

# float16으로 변환
embedding_weights_fp16 = compressed_weights.astype(np.float16)

print("임베딩 가중치 변환 완료:", embedding_weights_fp16.shape)

np.save("embedding_weights.npy", embedding_weights_fp16)  # Numpy 배열로 저장