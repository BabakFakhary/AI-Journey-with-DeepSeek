# به نام خدا                                                                  
from transformers import BartTokenizer, BartForConditionalGeneration

#===============================================================================
#                               Text Rewriting
#                                          بی حودی تلاش زیاد واسه درس عمل کردن نکنید چون برای  خلاصه سازی طراحی شده و ممکن هست درست عمل نکند
#===============================================================================

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def complete_text(partial_text, max_additional_length=50):
    """تکمیل متن ناتمام"""
    inputs = tokenizer(partial_text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        **inputs,
        max_length=len(partial_text.split()) + max_additional_length,
        num_beams=3,
        temperature=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# مثال
partial = "Once upon a time in a land far away,"
completed = complete_text(partial)
print(completed)