from transformers import AutoModel, AutoConfig

def print_model_architecture(model_name):
    # 모델 구성(Config) 로드
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # config.architectures 출력
    print(f"Model: {model_name}")
    print(f"Architectures: {config.architectures}")

    # 모델 데이터 타입 확인
    model_dtype = next(model.parameters()).dtype
    print(f"Model dtype: {model_dtype}")

    # 모델 출력
    print(model)
    

# 모델 확인
print_model_architecture("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

