from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# print(torch.__version__)  # PyTorch 버전 출력
# print(torch.cuda.is_available())  # GPU(CUDA) 사용 가능 여부 확인

# print(torch.backends.mps.is_available())  # MPS 지원 여부 확인
# print(torch.backends.mps.is_built())  # PyTorch가 MPS 지원 빌드인지 확인

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")

# # 간단한 텐서 연산 테스트
# x = torch.ones(3, 3).to(device)
# y = torch.ones(3, 3).to(device)
# z = x + y
# print(z)
# print(z.device)  # 출력: mps:0 -> 확인 됨

# ✅ 모델 정보
model_name = "deepseek-ai/deepseek-coder-6.7b-base"

# ✅ MPS 사용 가능 여부 확인
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.mps.empty_cache()  # MPS 메모리 정리

# ✅ 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ✅ 모델 로드 (`bfloat16` 유지)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # ✅ bfloat16 유지
).to(device)

# ✅ 질문 입력
input_text = "# BFS를 JavaScript 코드로 설명해줘."
inputs = tokenizer(input_text, return_tensors="pt")

# ✅ 입력 데이터를 MPS로 이동하되, `input_ids`는 반드시 `torch.long`으로 변환
inputs = {
    "input_ids": inputs["input_ids"].to(device, dtype=torch.long),  # ✅ torch.long 필수
    "attention_mask": inputs["attention_mask"].to(device, dtype=torch.bfloat16),  # ✅ bfloat16 유지
}

# ✅ 답변 생성
outputs = model.generate(
    **inputs, 
    max_new_tokens=50,  
    do_sample=True,  
    temperature=0.7,  
    top_p=0.9,  
    repetition_penalty=1.2  
)

# ✅ 최종 출력
print(tokenizer.decode(outputs[0], skip_special_tokens=True))