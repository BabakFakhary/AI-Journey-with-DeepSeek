#  به نام خدا                                                                           
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import nltk

# دانلود پکیج‌های مورد نیاز NLTK
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

from nltk.tokenize import sent_tokenize

# ==========================================================================================
#                                       Summarizer   
# ==========================================================================================

class BARTSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def chunk_text(self, text, max_chunk_size=1024):
        """تقسیم متن طولانی به بخش‌های کوچک‌تر"""
        try:
            sentences = sent_tokenize(text)
        except:
            # اگر NLTK کار نکرد، از تقسیم ساده استفاده کن
            sentences = text.split('. ')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + " " + sentence) < max_chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def summarize_long_text(self, text, max_length=150, min_length=40):
        """خلاصه‌سازی متن‌های بسیار طولانی"""
        chunks = self.chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            summary = self.summarize(chunk, max_length, min_length)
            summaries.append(summary)
        
        # اگر نیاز به خلاصه نهایی باشد
        if len(summaries) > 1:
            combined_summary = " ".join(summaries)
            return self.summarize(combined_summary, max_length, min_length)
        else:
            return summaries[0] if summaries else "No summary generated."
    
    def summarize(self, text, max_length=150, min_length=40):
        """خلاصه‌سازی متن معمولی"""
        inputs = self.tokenizer(
            [text],
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            early_stopping=True
        )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# استفاده عملی
summarizer = BARTSummarizer()

long_article = """
Artificial intelligence is revolutionizing the field of medicine in unprecedented ways. 
Recent advancements in machine learning algorithms have enabled computers to analyze medical images with accuracy that often surpasses human experts. 
In radiology, AI systems can detect early signs of diseases like cancer from X-rays and MRI scans. 
In pathology, they assist in analyzing tissue samples and identifying abnormalities. 
Beyond medical imaging, AI is transforming drug discovery by predicting how molecules will interact and identifying potential new treatments. 
Patient care is also being enhanced through AI-powered monitoring systems that can predict health deteriorations before they become critical. 
Additionally, personalized medicine is becoming a reality as AI algorithms analyze genetic data to recommend tailored treatment plans. 
While challenges remain regarding data privacy and regulatory approval, the integration of AI in healthcare promises to improve patient outcomes, reduce costs, and make quality care more accessible worldwide.
The future of healthcare is undoubtedly intertwined with the continued development and ethical implementation of artificial intelligence technologies.
"""

summary = summarizer.summarize_long_text(long_article)
print("📖 Original:")
print(long_article)
print("\n📝 Summarizer BART:")
print(summary)