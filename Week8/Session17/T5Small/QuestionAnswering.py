#                                                        به نام خدا
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# ===========================================================================
#  پاسخ به سوال (Question Answering)
# ===========================================================================

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def answer_question(question, context):
    """پاسخ به سوال بر اساس متن"""
    prompt = f"question: {question} context: {context}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_beams=3,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# مثال عملی: سیستم پرسش و پاسخ
context = """
Microsoft Corporation is an American multinational technology company. 
It develops, manufactures, licenses, supports, and sells computer software, consumer electronics, personal computers, and related services. 
Its best known software products are the Microsoft Windows line of operating systems, the Microsoft Office suite, and the Internet Explorer and Edge web browsers. 
The company was founded by Bill Gates and Paul Allen on April 4, 1975.
"""

questions = [
    "Who founded Microsoft?",
    "What are Microsoft's famous products?",
    "When was Microsoft founded?"
]

print("❓ Question and Answer system:")
for question in questions:
    answer = answer_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("-" * 50)