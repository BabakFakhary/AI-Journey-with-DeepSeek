#                                                                        به نام خدا                                                                  
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

# ==================================================================================
#                                      استفاده از مدل ParsBERT                             
# ==================================================================================
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PersianBART:
    def __init__(self):
        # مدل مبتنی بر BART برای فارسی
        # استفاده از مدل‌های معتبر
        self.model_name = "persiannlp/mt5-small-parsinlu-opus-translation_fa_en"
        # یا
        # self.model_name = "Viraa/parsT5-summary"
        # یا
        # self.model_name = "m3hrdadfi/mt5-small-persian-summarization"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    
    def summarize(self, text, max_length=100):
        # برای خلاصه‌سازی ساده
        prompt = f"خلاصه کن: {text}"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# تست
persian_text = """
هوش مصنوعی در حال تحول صنعت پزشکی است. پیشرفت‌های اخیر در الگوریتم‌های یادگیری ماشین 
توانایی کامپیوترها را در تحلیل تصاویر پزشکی با دقتی فراتر از متخصصان انسانی فراهم کرده است.
"""

summarizer = PersianBART()
summary = summarizer.summarize(persian_text)
print(fa("📖 متن اصلی:"))
print(fa(persian_text))
print(fa("\n📝 خلاصه فارسی:"))
print(fa(summary))