from transformers import AutoModel, AutoConfig, AutoTokenizer

"""
WordPiece - BertTokenizer,	DistilBertTokenizer
SentencePiece - T5Tokenizer, XLNetTokenizer, AlbertTokenizer
BPE - RobertaTokenizer, GPT2Tokenizer
Byte-Level BPE - OpenAI GPT2, Llama
"""
def print_model_architecture(model_name):
    # 모델 구성(Config) 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # config.architectures 출력
    print(f"Model: {model_name}")
    print(f"Architectures: {config.architectures}")

    # 모델 데이터 타입 확인
    model_dtype = next(model.parameters()).dtype
    print(f"Model dtype: {model_dtype}")

    # 토크나이저 종류 출력 : BPE/WordPiece / SentencePiece 등
    print(f"Tokenizer:  {tokenizer.__class__}")
    print(tokenizer)

    # 모델 출력
    print(model)
    

# 모델 확인
print_model_architecture("skt/kogpt2-base-v2")
#print_model_architecture("jhgan/ko-sbert-nli")

