import json
from transformers import AutoTokenizer

#MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODEL_NAME = "skt/kogpt2-base-v2"

def save_mapping_json(vocab, filename="mapping.json"):
    """모든 토큰과 해당 ID를 JSON 파일로 저장 (\r, \n, \t 제거)"""
    cleaned_count = 0  # 제어 문자 제거된 토큰 수 카운트
    
    # 토큰과 ID를 딕셔너리로 저장
    mapping = {}
    for token, idx in vocab.items():
        cleaned_token = token.replace("\r", "").replace("\n", "").replace("\t", "")
        if cleaned_token != token:
            cleaned_count += 1
        
        mapping[cleaned_token] = idx

    # JSON 파일로 저장
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)
    
    print(f"Mapping saved to {filename}")
    print(f"총 {len(vocab)}개의 토큰 저장, {cleaned_count}개 토큰에서 제어문자 제거됨.")


def load_mapping_json(filename="mapping.json"):
    """JSON 파일을 로드하여 딕셔너리로 반환"""
    with open(filename, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    print(f"Mapping loaded from {filename}")
    return {token: int(idx) for token, idx in mapping.items()}

# KoGPT2 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 모든 토큰과 ID 가져오기
vocab = tokenizer.get_vocab()  # 모델의 전체 단어 사전 가져오기

# JSON 매핑 파일 저장
save_mapping_json(vocab, "mapping.json")

# JSON 매핑 파일 로드
loaded_mapping = load_mapping_json("mapping.json")

# 로드된 매핑 일부 확인 (상위 10개)
for token, idx in list(loaded_mapping.items())[:10]:
    print(f"Token: {token} -> ID: {idx}")
