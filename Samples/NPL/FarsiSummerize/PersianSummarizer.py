#                                                                      به نام خدا                                                                  
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
#                                       استفاده از مدل های مخصوص فارسی                             
# ==================================================================================

class PersianSummarizer:
    def __init__(self):
        # استفاده از مدل مناسب برای خلاصه‌سازی فارسی
        self.model_name = "google/mt5-small"
        # m3hrdadfi/mt5-small-persian-summarization
        # یا از این مدل استفاده کنید: "google/mt5-small"
        
        try:
            print(fa("📥 در حال بارگذاری مدل خلاصه‌سازی..."))
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            print(fa("✅ مدل با موفقیت بارگذاری شد"))
        except Exception as e:
            print(fa(f"❌ خطا در بارگذاری مدل: {e}"))
            print(fa("🔧 در حال استفاده از مدل جایگزین..."))
            self.model_name = "google/mt5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    
    def summarize(self, text, max_length=150, min_length=40):
        try:
            # اضافه کردن پیشوند برای خلاصه‌سازی
            if "mt5" in self.model_name.lower():
                input_text = "خلاصه کن: " + text
            else:
                input_text = "summarize: " + text
            
            # توکنایز کردن متن ورودی
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length"
            )
            
            # تولید خلاصه
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=2.0,
                length_penalty=1.0,
                no_repeat_ngram_size=3
            )
            
            # دیکد کردن خروجی
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            return fa(f"خطا در خلاصه‌سازی: {str(e)}")

# تست با متن طولانی‌تر
persian_text = """
هوش مصنوعی در حال تحول صنعت پزشکی است. پیشرفت‌های اخیر در الگوریتم‌های یادگیری ماشین 
توانایی کامپیوترها را در تحلیل تصاویر پزشکی با دقتی فراتر از متخصصان انسانی فراهم کرده است. 
در رادیولوژی، سیستم‌های هوش مصنوعی می‌توانند علائم اولیه بیماری‌هایی مانند سرطان را در 
عکس‌های اشعه ایکس و اسکن MRI تشخیص دهند. این سیستم‌ها قادرند الگوهایی را شناسایی کنند 
که حتی برای چشم انسان قابل مشاهده نیستند. در پاتولوژی، هوش مصنوعی در تحلیل نمونه‌های بافت 
و شناسایی ناهنجاری‌ها کمک می‌کند. علاوه بر تصویربرداری پزشکی، هوش مصنوعی در حال 
تحول کشف داروها با پیش‌بینی نحوه تعامل مولکول‌ها و شناسایی درمان‌های بالقوه جدید است.
این فناوری می‌تواند زمان و هزینه مورد نیاز برای توسعه داروهای جدید را به میزان قابل توجهی کاهش دهد.
همچنین در زمینه پزشکی شخصی، هوش مصنوعی می‌تواند درمان‌های سفارشی بر اساس ژنتیک و 
سوابق پزشکی هر بیمار ارائه دهد.
"""

summarizer = PersianSummarizer()
summary = summarizer.summarize(persian_text, max_length=100, min_length=30)

print(fa("📖 متن اصلی:"))
print(fa(persian_text))
print(fa("\n📝 خلاصه فارسی:"))
print(fa(summary))