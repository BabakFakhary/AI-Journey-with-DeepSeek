#                                                                      به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
from transformers import pipeline
import torch

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

#============================================================================
#  Qusetion Anwser With Hugging Face Transformer , Customize For Farsi
#============================================================================

# مدل‌های تست شده و کارآمد
MODEL_OPTIONS = [
    "bert-large-uncased-whole-word-masking-finetuned-squad",  # مدل استاندارد انگلیسی
    "distilbert-base-cased-distilled-squad",  # مدل سبک
    "deepset/roberta-base-squad2",  # مدل محبوب
]

for model_name in MODEL_OPTIONS:
    try:
        print(fa(f"در حال بارگذاری مدل: {model_name}"))
        qa_pipeline = pipeline("question-answering", model=model_name)
        print(fa(f"✅ مدل {model_name} بارگذاری شد"))
        break
    except Exception as e:
        print(fa(f"خطا در بارگذاری {model_name}: {e}"))
else:
    print(fa("هیچ مدلی بارگذاری نشد، از مدل پیش‌فرض استفاده می‌کنیم..."))
    qa_pipeline = pipeline("question-answering")

# متن زمینه به فارسی
context = """
شرکت هوش مصنوعی دانش بنیان آرمان در سال ۱۳۹۵ در تهران تأسیس شد. 
این شرکت در حوزه پردازش زبان فارسی و بینایی کامپیوتر فعالیت می‌کند.
تعداد کارمندان شرکت ۵۰ نفر است که شامل ۳۵ مهندس نرم‌افزار و ۱۵ متخصص داده می‌باشد.
محصول اصلی شرکت یک دستیار هوشمند فارسی به نام "پارسا" است.
مدیرعامل شرکت دکتر علی رضایی است که دکترای خود را از دانشگاه شریف گرفته است.
"""

# سوالات به فارسی
questions = [
    "شرکت کی تأسیس شد؟",
    "تعداد کارمندان چند نفر است؟",
    "محصول اصلی شرکت چیست؟",
    "مدیرعامل شرکت چه کسی است؟",
    "شرکت در چه حوزه‌هایی فعالیت می‌کند؟"
]

print(fa("🔍 سیستم پرسش و پاسخ فارسی"))
print()
print(fa("متن زمینه:"))
print(fa(context[:200] + "..."))
print("\n" + "="*50)

for question in questions:
    try:
        result = qa_pipeline(question=question, context=context)
        print(fa(f"❓ سوال: {question}"))
        print(fa(f"✅ پاسخ: {result['answer']}"))
        print(fa(f"🎯 امتیاز: {result['score']:.3f}"))
        print("-" * 50)
    except Exception as e:
        print(fa(f"خطا در پردازش سوال: {e}"))