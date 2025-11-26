# به نام خدا                                                                           
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

# ============================================================================
#                                      خلاصه سازی                                   
# ============================================================================
class PegasusSummarizer:
    def __init__(self, model_name="google/pegasus-xsum"):
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def summarize(self, text, max_length=150, min_length=40):
        """خلاصه‌سازی متن"""
        # توکنایز کردن متن
        inputs = self.tokenizer(
            text, 
            max_length=1024, 
            truncation=True, 
            padding="longest", 
            return_tensors="pt"
        ).to(self.device)
        
        # تولید خلاصه
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        
        # دیکد کردن خلاصه
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

# استفاده عملی
summarizer = PegasusSummarizer()

text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.
Leading AI textbooks define the field as the study of intelligent agents: any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
Some popular accounts use the term artificial intelligence to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving.
AI applications include advanced web search engines (e.g., Google), recommendation systems (used by YouTube, Amazon and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Tesla),
automated decision-making and competing at the highest level in strategic game systems (such as chess and Go). As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI,
a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
"""

print("\n" + "="*70)
summary = summarizer.summarize(text)
print("Orginal :", len(text), "Character")
print("summary:", summary)
print("="*70)