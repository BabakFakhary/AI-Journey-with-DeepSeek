#                                                                        به نام خدا
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# ===========================================================================
#  خلاصه‌سازی متن (Text Summarization)
# ===========================================================================

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize_text(text):
    """خلاصه‌سازی متن‌های طولانی"""
    prompt = f"summarize: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# مثال عملی: خلاصه‌سازی مقاله
article = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
Leading AI textbooks define the field as the study of intelligent agents: any system that perceives its environment and takes actions that maximize its chance of achieving its goals. 
AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making and competing at the highest level in strategic game systems. 
The field was founded on the assumption that human intelligence can be so precisely described that a machine can be made to simulate it.
"""

summary = summarize_text(article)
print("📝 summarize : ")
print(summary)