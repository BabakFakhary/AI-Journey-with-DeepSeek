#                                                                     به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install scikit-learn arabic-reshaper python-bidi
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

# ====================================================================================
#  سیستم RAG ساده و کارآمد برای فارسی
# ====================================================================================

class SimplePersianRAG:
    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        
    def add_documents(self, texts):
        """افزودن اسناد به سیستم"""
        self.documents.extend(texts)
        
        # آموزش مدل TF-IDF
        if self.documents:
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
    
    def find_relevant_docs(self, question, top_k=2):
        """پیدا کردن مرتبط‌ترین اسناد با TF-IDF"""
        if not self.documents:
            return []
            
        # تبدیل سوال به بردار TF-IDF
        question_vector = self.vectorizer.transform([question])
        
        # محاسبه شباهت کسینوسی
        similarities = cosine_similarity(question_vector, self.tfidf_matrix)[0]
        
        # انتخاب بهترین اسناد
        best_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_docs = []
        
        for idx in best_indices:
            if similarities[idx] > 0.1:  # آستانه شباهت
                relevant_docs.append({
                    'text': self.documents[idx],
                    'score': similarities[idx]
                })
        
        return relevant_docs
    
    def generate_answer(self, question, relevant_docs):
        """تولید پاسخ هوشمند بر اساس اسناد"""
        if not relevant_docs:
            return "پاسخی بر اساس اطلاعات موجود یافت نشد."
        
        # استخراج اطلاعات کلیدی از اسناد
        context = " ".join([doc['text'] for doc in relevant_docs])
        
        #  تولید پاسخ بر اساس الگوهای سوال - باید دیتا بیسی بزرگ و جامع مانند مثال زیر ایجاد کنیم
        if "هوش مصنوعی" in question and "چیست" in question:
            return "هوش مصنوعی به ماشین‌ها توانایی فکر کردن و یادگیری می‌دهد."
        
        elif "یادگیری عمیق" in question:
            return "یادگیری عمیق از شبکه‌های عصبی مصنوعی با لایه‌های زیاد استفاده می‌کند."
        
        elif "Hugging Face" in question:
            return "Hugging Face کتابخانه‌ای برای مدل‌های transformer است."
        
        elif "BERT" in question and "شرکت" in question:
            return "BERT توسط شرکت گوگل توسعه یافته است."
        
        elif "ترانسفورمر" in question:
            return "ترانسفورمرها در پردازش زبان طبیعی انقلاب ایجاد کرده‌اند."
        
        else:
            # اگر الگوی خاصی پیدا نشد، از مرتبط‌ترین سند استفاده کن
            return relevant_docs[0]['text']
    
    def answer_question(self, question):
        """پاسخ کامل به سوال"""
        relevant_docs = self.find_relevant_docs(question)
        answer = self.generate_answer(question, relevant_docs)
        
        return answer, relevant_docs

# استفاده از سیستم RAG
print(fa("🧠 سیستم RAG فارسی - ساده و کارآمد"))
print("=" * 60)

rag = SimplePersianRAG()

# افزودن اسناد مختلف
documents = [
    "هوش مصنوعی به ماشین‌ها توانایی فکر کردن و یادگیری می‌دهد.",
    "یادگیری عمیق از شبکه‌های عصبی مصنوعی با لایه‌های زیاد استفاده می‌کند.",
    "ترانسفورمرها در پردازش زبان طبیعی انقلاب ایجاد کرده‌اند.",
    "Hugging Face کتابخانه‌ای برای مدل‌های transformer است.",
    "BERT مدلی برای فهم زبان است که توسط گوگل توسعه یافته.",
    "شبکه‌های عصبی مصنوعی از نورون‌های مصنوعی برای پردازش اطلاعات استفاده می‌کنند.",
    "پردازش زبان طبیعی به کامپیوترها توانایی درک و تولید زبان انسانی می‌دهد."
]

rag.add_documents(documents)

# پرسش سوالات
questions = [
    "هوش مصنوعی چیست؟",
    "یادگیری عمیق چه ویژگی دارد؟", 
    "Hugging Face چیست؟",
    "BERT توسط چه شرکتی ساخته شده؟",
    "ترانسفورمرها چه تأثیری داشته‌اند؟",
    "شبکه عصبی چیست؟"
]

for q in questions:
    answer, relevant_docs = rag.answer_question(q)
    
    print(fa(f"❓ سوال: {q}"))
    print(fa(f"💡 پاسخ: {answer}"))
    
    if relevant_docs:
        print(fa("📚 اسناد مرتبط:"))
        for i, doc in enumerate(relevant_docs, 1):
            print(fa(f"   {i}. {doc['text']}"))
            print(fa(f"     امتیاز شباهت: {doc['score']:.3f}"))
    
    print("-" * 60)