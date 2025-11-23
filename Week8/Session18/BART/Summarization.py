#                                                                به نام خدا                                                               
from transformers import BartTokenizer, BartForConditionalGeneration

#=====================================================================
#                             BART Summarization
#=====================================================================
from transformers import BartTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def summarize_text(text, max_length=130, min_length=30):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# مثال عملی
article = """
Artificial intelligence is transforming healthcare in numerous ways. 
AI algorithms can now analyze medical images with accuracy surpassing human experts. 
They help in drug discovery, patient monitoring, and personalized treatment plans. 
The integration of AI in healthcare promises to improve outcomes and reduce costs.
"""

summary = summarize_text(article)
print("📝 Summarization:")
print(summary)