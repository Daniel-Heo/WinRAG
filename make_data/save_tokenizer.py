from transformers import AutoTokenizer
import os

MODEL_NAME = "skt/kogpt2-base-v2"
#MODEL_NAME = "jhgan/ko-sbert-nli"

def save_only_tokenizer_json(src_model: str, tgt_directory: str):
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(src_model)

    # 저장할 디렉터리 경로 설정
    tgt_directory = os.path.abspath(tgt_directory)  # 절대 경로 변환
    os.makedirs(tgt_directory, exist_ok=True)  # 디렉터리 생성 (존재하지 않으면)

    # tokenizer.json 파일만 저장
    tokenizer_path = os.path.join(tgt_directory, "tokenizer.json")
    tokenizer.backend_tokenizer.save(tokenizer_path)

    print(f"tokenizer.json saved at: {tokenizer_path}")

# 실행
save_only_tokenizer_json(MODEL_NAME, "./")
