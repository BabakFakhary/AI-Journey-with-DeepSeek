#                                                                     به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------

from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

# ====================================================================================
#  سیستم RAG (Retrieval-Augmented Generation)
# ====================================================================================

from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RAGSystem:
    def __init__(self):
        # مدل برای embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L12-v2')
        
        # مدل برای تولید پاسخ
        self.generator = pipeline(
            "text2text-generation",
            model="google/mt5-small",
            tokenizer="google/mt5-small"
        )
        
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, texts):
        """افزودن اسناد به سیستم"""
        self.documents.extend(texts)
        
        # ایجاد embeddings برای اسناد
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()
            self.embeddings.append(embedding)
    
    def find_relevant_docs(self, question, top_k=3):
        """پیدا کردن مرتبط‌ترین اسناد"""
        # ایجاد embedding برای سوال
        inputs = self.tokenizer(question, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        question_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        
        # محاسبه شباهت
        similarities = []
        for doc_embedding in self.embeddings:
            similarity = cosine_similarity(question_embedding, doc_embedding)[0][0]
            similarities.append(similarity)
        
        # انتخاب بهترین اسناد
        best_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in best_indices]
    
    def answer_question(self, question):
        """پاسخ به سوال بر اساس اسناد"""
        relevant_docs = self.find_relevant_docs(question)
        context = " ".join(relevant_docs)
        
        prompt = f"بر اساس متن زیر به سوال پاسخ دهید:\n{context}\n\nسوال: {question}\nپاسخ:"
        
        response = self.generator(
            prompt,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7
        )
        
        return response[0]['generated_text']

# استفاده از سیستم RAG
rag = RAGSystem()

# افزودن اسناد مختلف
documents = [
    "هوش مصنوعی به ماشین‌ها توانایی فکر کردن و یادگیری می‌دهد.",
    "یادگیری عمیق از شبکه‌های عصبی مصنوعی با لایه‌های زیاد استفاده می‌کند.",
    "ترانسفورمرها در پردازش زبان طبیعی انقلاب ایجاد کرده‌اند.",
    "Hugging Face کتابخانه‌ای برای مدل‌های transformer است.",
    "BERT مدلی برای فهم زبان است که توسط گوگل توسعه یافته."
]

rag.add_documents(documents)

# پرسش سوالات
questions = [
    "هوش مصنوعی چیست؟",
    "یادگیری عمیق چه ویژگی دارد؟",
    "Hugging Face چیست؟",
    "BERT توسط چه شرکتی ساخته شده؟"
]

for q in questions:
    answer = rag.answer_question(q)
    print(fa(f"❓ {q}"))
    print(fa(f"💡 {answer}\n"))