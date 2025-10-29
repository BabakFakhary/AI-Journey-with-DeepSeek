#                                                                    به نام خدا
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# ===========================================================================
#  تشخیص احساسات (Sentiment Analysis)
# ===========================================================================

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def analyze_sentiment(text):
    """تحلیل احساسات متن"""
    prompt = f"sentiment: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=10,
            num_beams=2,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# مثال عملی: تحلیل نظرات کاربران
reviews = [
    "I love this product! It's amazing and works perfectly.",
    "This is the worst service I have ever experienced.",
    "The movie was okay, nothing special but not bad either."
]

print("😊 Sentiment Analysis:")
for review in reviews:
    sentiment = analyze_sentiment(review)
    print(f"Opinion: {review}")
    print(f"Feeling: {sentiment}")
    print("-" * 50)