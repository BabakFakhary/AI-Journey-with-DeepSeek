#                                                                        به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

#============================================================================
#  چت بات هوشمند با حافظه
#============================================================================

class SmartChatbot:
    def __init__(self):
        self.conversation_history = []
        
        # بارگذاری مدل گفتگو
        self.chat_pipeline = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            tokenizer="microsoft/DialoGPT-medium"
        )
    
    def add_context(self, context_text):
        """افزودن متن زمینه به حافظه"""
        self.conversation_history.append(f"متن زمینه: {context_text}")
    
    def ask_question(self, question):
        """پرسش سوال بر اساس زمینه"""
        # ساخت prompt از تاریخچه و سوال جدید
        prompt = "\n".join(self.conversation_history) + f"\nسوال: {question}\nپاسخ:"
        
        # تولید پاسخ
        response = self.chat_pipeline(
            prompt,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=50256
        )
        
        answer = response[0]['generated_text'].split("پاسخ:")[-1].strip()
        
        # ذخیره در تاریخچه
        self.conversation_history.append(f"سوال: {question}")
        self.conversation_history.append(f"پاسخ: {answer}")
        
        return answer

# استفاده از چت بات
bot = SmartChatbot()

# افزودن متن زمینه
context = """
شرکت ما در زمینه هوش مصنوعی فعالیت می‌کند. 
ما محصولات متنوعی داریم از جمله دستیار صوتی، سیستم پیشنهاد دهنده و تحلیل احساسات.
تیم ما متشکل از ۱۰ مهندس داده و ۵ متخصص یادگیری ماشین است.
"""

bot.add_context(context)

# پرسش سوالات
questions = [
    "شرکت شما چه کاری می‌کند؟",
    "تعداد کارمندان شما چقدر است؟",
    "چه محصولاتی ارائه می‌دهید؟"
]

for q in questions:
    answer = bot.ask_question(q)
    print(fa(f"👤: {q}"))
    print(fa(f"🤖: {answer}\n"))