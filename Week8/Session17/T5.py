#                                                                       به نام خدا 
# pip install sentencepiece
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# ==================================================================================
#  پیاده سازی T5
# توجه : 
#    مدل T5-Small برای ترجمه آموزش ندیده است
# ==================================================================================

# بارگذاری مدل و توکنایزر
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def t5_translate(text, source_lang="en", target_lang="fr"):
    """ترجمه با T5"""
    prompt = f"translate {source_lang} to {target_lang}: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# تست ترجمه
english_text = "Hello World!"
french_translation = t5_translate(english_text, "en", "fr")

# -------------------------------------------------------------
# خروجی با ورودی یکسان خواهد بود پس بدنبال ترجمه نباشد
# -------------------------------------------------------------
print(f"English: {english_text}")
print(f"French: {french_translation}")