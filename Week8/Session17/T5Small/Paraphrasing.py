#                                                                  به نام خدا

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# ===========================================================================
#  بازنویسی و پارافریز متن (Paraphrasing)
# ===========================================================================

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def paraphrase_text(text):
    """بازنویسی متن با حفظ معنی"""
    prompt = f"paraphrase: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_beams=3,
            temperature=0.8,
            do_sample=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# مثال عملی: بازنویسی جملات
original_sentences = [
    "The weather is very nice today",
    "I really enjoy learning about machine learning",
    "This product is very good and high quality"
]

print("✍️ Rewriting the text:")
for sentence in original_sentences:
    paraphrased = paraphrase_text(sentence)
    print(f"Orginal: {sentence}")
    print(f"Rewriting: {paraphrased}")
    print("-" * 50)