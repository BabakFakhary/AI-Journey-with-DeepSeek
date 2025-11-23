#    به نام خدا                                                                             
# استفاده از مدلی که مخصوص ترجمه آموزش دیده
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

#===============================================================
#                            T5
#===============================================================

# بارگذاری مدل و توکنایزر
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def t5_translate_fixed(text, source_lang="en", target_lang="fr"):
    """ترجمه با T5 - نسخه بهبود یافته"""
    # فرمت ساده‌تر
    prompt = f"translate {source_lang} to {target_lang}: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            early_stopping=True,
            num_return_sequences=1
        )
    
    # پاکسازی خروجی
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # حذف احتمالی prefix تکراری
    if translated_text.startswith(f"translate {source_lang} to {target_lang}:"):
        translated_text = translated_text.replace(f"translate {source_lang} to {target_lang}:", "").strip()
    
    return translated_text

# استفاده از مدل بزرگتر که نتایج بهتری می‌دهد
def t5_large_translate(text, source_lang="en", target_lang="fr"):
    """ترجمه با T5-large"""
    tokenizer_large = T5Tokenizer.from_pretrained("t5-large")
    model_large = T5ForConditionalGeneration.from_pretrained("t5-large")
    
    prompt = f"translate English to French: {text}"
    inputs = tokenizer_large(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model_large.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
    
    return tokenizer_large.decode(outputs[0], skip_special_tokens=True)

# تست
english_text = "Hello, how are you today?"
french_translation = t5_translate_fixed(english_text)
print(f"English: {english_text}")
print(f"French: {french_translation}")

# تست large
english_text = "Hello, how are you today?"
french_translation = t5_large_translate(english_text)
print(f"English: {english_text}")
print(f"French: {french_translation}")