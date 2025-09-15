#                                                                            به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# ابتدا کتابخانه را نصب کنید (در ترمینال): pip install transformers
from transformers import pipeline

# ============================================================================================
#                            یک مثال ساده کد (برای تحلیل احساسات)
# ============================================================================================

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

# یک pipeline برای تحلیل احساسات ایجاد می‌کنیم.
# کتابخانه به طور خودکار مدل پیش‌فرض مناسب را دانلود می‌کند.
classifier = pipeline('sentiment-analysis')

# متن مورد نظر را می‌دهیم و نتیجه را می‌گیریم.
result = classifier('I love this product! The quality is amazing.')

print(result)
# خروجی: [{'label': 'POSITIVE', 'score': 0.9998}]