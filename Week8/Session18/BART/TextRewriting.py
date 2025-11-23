# به نام خدا                                                                  
from transformers import BartTokenizer, BartForConditionalGeneration

#===============================================================================
#                               Text Rewriting
#                                          بی حودی تلاش زیاد واسه درس عمل کردن نکنید چون برای  خلاصه سازی طراحی شده و ممکن هست درست عمل نکند
#===============================================================================

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def paraphrase_text(text):
    """بازنویسی متن با حفظ معنی"""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        **inputs,
        max_length=len(text.split()) + 10,
        num_beams=5,
        temperature=0.8,
        do_sample=True,
        repetition_penalty=1.2
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# مثال
original = "The weather is very nice today"
paraphrased = paraphrase_text(original)
print(f"Original: {original}")
print(f"Rewriting: {paraphrased}")