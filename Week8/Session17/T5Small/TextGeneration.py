#                                                                    به نام خدا
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# ===========================================================================
#  تولید متن (Text Generation)
# ===========================================================================

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def generate_text(prompt, max_length=100):
    """تولید متن بر اساس پرامپت"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=3,
            temperature=0.9,
            do_sample=True,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# مثال عملی: تولید متن خلاق
prompts = [
    "The future of artificial intelligence",
    "How to learn programming",
    "The benefits of renewable energy"
]

print("🧠 Creative text production:")
for prompt in prompts:
    generated = generate_text(prompt)
    print(f"Prompt:  {prompt}")
    print(f"Creative:  {generated}")
    print("-" * 50)