#                                                                       به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
from transformers import pipeline

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

#============================================================================
#  Qusetion Anwser With Hugging Face Transformer
#============================================================================

# ایجاد pipeline برای پرسش و پاسخ
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    tokenizer="distilbert-base-cased-distilled-squad"
)

# متن زمینه
context = """
هوش مصنوعی شاخه‌ای از علوم کامپیوتر است که به ساخت ماشین‌های هوشمند می‌پردازد.
یادگیری عمیق زیرشاخه‌ای از یادگیری ماشین است که از شبکه‌های عصبی استفاده می‌کند.
ترانسفورمرها معماری‌هایی هستند که در پردازش زبان طبیعی استفاده می‌شوند.
"""

# سوالات
questions = [
    "هوش مصنوعی چیست؟",
    "یادگیری عمیق چه رابطه‌ای با یادگیری ماشین دارد؟",
    "ترانسفورمرها در چه حوزه‌ای استفاده می‌شوند؟"
]

for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(fa(f"سوال: {question}"))
    print(fa(f"پاسخ: {result['answer']}"))
    print(fa(f"امتیاز: {result['score']:.3f}"))
    print("-" * 50)