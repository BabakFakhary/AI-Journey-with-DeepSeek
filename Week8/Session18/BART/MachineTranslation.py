# به نام خدا                                                                  
from transformers import BartTokenizer, BartForConditionalGeneration

#===============================================================================
#                               Machine Translation
#                                          بی حودی تلاش زیاد واسه درس عمل کردن نکنید چون برای  خلاصه سازی طراحی شده و ممکن هست درست عمل نکند 
#===============================================================================

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def translate_with_bart(text, source_lang="en", target_lang="fr"):
    # BART می‌تواند برای ترجمه فاین-تون شود
    prompt = f"translate {source_lang} to {target_lang}: {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# تست
english_text = "Hello, how are you today?"
french_translation = translate_with_bart(english_text)
print(f"English: {english_text}")
print(f"French: {french_translation}") 