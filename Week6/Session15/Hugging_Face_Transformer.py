#                                                                     به نام خدا   
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install transformers datasets accelerate
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    pipeline,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import warnings

# ============================================================================================
#                             Multiple Tasks با Hugging Face Transformers  - Tasks :
# 1.Token Classification (NER)
# 2.Question Answering
# 3.Text Generation
# 4.مثال عملی سیستم کامل NLP 
# 5.Optimization و Deployment
# ============================================================================================

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

# -------------------------------------------------------

warnings.filterwarnings('ignore')

# بررسی GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"="*50)
print(f"Using device: {device}")
print(f"="*50)

# -------------------------------------------------------
# معرفی مدل‌های چندمنظوره
#  ------------------------------------------------------

# مدل‌های پشتیبانی شده برای تسک‌های مختلف
MODELS = {
    'classification': 'bert-base-uncased',
    'ner': 'dslim/bert-base-NER',
    'qa': 'bert-large-uncased-whole-word-masking-finetuned-squad',
    'generation': 'gpt2'
}

# مدل‌های پشتیبانی شده برای تسک‌های مختلف
MODELS = {
    'classification': 'bert-base-uncased',
    'ner': 'dslim/bert-base-NER',
    'qa': 'bert-large-uncased-whole-word-masking-finetuned-squad',
    'generation': 'gpt2'
}

print("Available models for different tasks:")
for task, model_name in MODELS.items():
    print(f"{task.upper():15}: {model_name}")
print(f"="*50)

# =====================================================
# 1. Token Classification (NER)
# =====================================================

# داده نمونه برای NER
# این یک نمونه داده استاندارد برای آموزش یا ارزیابی یک مدل تشخیص موجودیت‌های نامدار (NER) است.
  # بخش  داخل  entities :
    # "start": 0 - شروع موجودیت در رشته متن است. A in Apple
    # "end": 5 - این شاخص پایان موجودیت است. e in Apple (َطول)
    # "label": "ORG" - این برچسب یا نوع موجودیت است
    # شاید برای شما این سوال پیش بیاید که چرا در این جمله، موجودیت‌های دیگر مانند
      # "Berlin" (که یک مکان یا LOC است)
      # برچسب نخورده‌اند. این می‌تواند چند دلیل داشته باشد
        # ین داده ممکن است فقط برای آموزش تشخیص موجودیت ORG در نظر گرفته شده باشد
        # ممکن است این یک نمونه تمرینی ساده باشد و در dataset کامل، تمام موجودیت‌ها به طور کامل برچسب‌گذاری شوند.
        # ممکن است برچسب‌گذار انسان تنها روی Apple به عنوان یک سازمان تمرکز کرده و بقیه را نادیده گرفته باشد (که یک خطا در برچسب‌گذاری است).
ner_data = [
    {"text": "Apple is looking to buy a startup in Berlin for $1 billion", "entities": [{"start": 0, "end": 5, "label": "ORG"}]},
    {"text": "Elon Musk founded SpaceX in California", "entities": [{"start": 0, "end": 9, "label": "PER"}, {"start": 18, "end": 24, "label": "ORG"}]},
    {"text": "I visited Paris and London last summer", "entities": [{"start": 10, "end": 15, "label": "LOC"}, {"start": 20, "end": 26, "label": "LOC"}]}
]

# ایجاد DataFrame
ner_df = pd.DataFrame(ner_data)
print("NER Dataset:")
print(ner_df)

# -----------------------------------------------------
#  استفاده از Pipeline برای NER
# -----------------------------------------------------

# ایجاد NER pipeline
ner_pipeline = pipeline(
    "token-classification",
    model=MODELS['ner'],
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

# تست NER
text = "Apple is opening a new store in San Francisco next month"
entities = ner_pipeline(text)

print(f"="*50)
print("\nNER ---->")
print(f"Text: {text}")
print("Named Entities:")
for entity in entities:
    # خروجی  تحلیل
      # چرا کلمات دیگر شناسایی نشدند
        # کلمات دیگری مانند "store"، "month" یا "new" اسامی خاص نیستند. آنها جزو موجودیت‌های نامدار محسوب نمی‌شوند. موجودیت‌های نامدار معمولاً به اسامی افراد، مکان‌ها، سازمان‌ها، تاریخ‌های خاص و مقادیر پولی خاص محدود می‌شوند
          # اگر طول رشته کمتر از 15 کاراکتر باشد، پایتون به اندازه لازم space (فاصله) به سمت راست آن اضافه میکند تا طول کل به 15 برسد. 
    print(f"  {entity['word']:15} -> {entity['entity_group']} (confidence: {entity['score']:.3f})")
print(f"="*50)

# -----------------------------------------------------
# 1-1. پیاده‌سازی دقیق‌تر NER
# -----------------------------------------------------

# بارگذاری مدل و توکنایزر برای NER
ner_tokenizer = AutoTokenizer.from_pretrained(MODELS['ner'])
ner_model = AutoModelForTokenClassification.from_pretrained(MODELS['ner'])
ner_model.to(device)

def advanced_ner(text):
    inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = ner_model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    entities = []
    current_entity = {"word": "", "label": "", "start": 0, "end": 0}
    
    for i, (token, prediction) in enumerate(zip(tokens, predictions[0])):
        label = ner_model.config.id2label[prediction.item()]
        
        if label != "O":
            entity_type = label.split("-")[1]
            if label.startswith("B-"):
                if current_entity["word"]:
                    entities.append(current_entity)
                current_entity = {"word": token, "label": entity_type, "start": i, "end": i}
            elif label.startswith("I-"):
                current_entity["word"] += " " + token
                current_entity["end"] = i
        elif current_entity["word"]:
            entities.append(current_entity)
            current_entity = {"word": "", "label": "", "start": 0, "end": 0}
    
    if current_entity["word"]:
        entities.append(current_entity)
    
    # تمیز کردن tokens
    for entity in entities:
        entity["word"] = entity["word"].replace(" ##", "")
    
    return entities

# تست NER پیشرفته
text = "Microsoft was founded by Bill Gates in Albuquerque"
entities = advanced_ner(text)
print("\nAdvanced NER ---->")
print(f"Text: {text}")
print("Named Entities:")
for entity in entities:
    print(f"  {entity['word']:15} -> {entity['label']}")
print(f"="*50)

# =====================================================
# 2.  Question Answering
# =====================================================

# آماده‌سازی داده برای QA
# داده نمونه برای Question Answering
qa_data = [
    {
        "context": "The Amazon rainforest is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America.",
        "question": "Where is the Amazon rainforest located?",
        "answers": {"text": ["Amazon basin of South America"], "answer_start": [87]}
    },
    {
        "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "question": "In which city is the Eiffel Tower located?",
        "answers": {"text": ["Paris"], "answer_start": [58]}
    }
]

print("QA Dataset Sample:")
for i, item in enumerate(qa_data):
    print(f"{i+1}. Context: {item['context'][:50]}...")
    print(f"   Question: {item['question']}")
    print(f"   Answer: {item['answers']['text'][0]}")
    print()    
print(f"="*50)

# -----------------------------------------------------
#  استفاده از QA Pipeline
# -----------------------------------------------------
# ایجاد QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model=MODELS['qa'],
    device=0 if torch.cuda.is_available() else -1
)

# تست QA
context = "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China to protect the Chinese states and empires against the raids and invasions of the various nomadic groups of the Eurasian Steppe."
question = "What materials were used to build the Great Wall?"

result = qa_pipeline(question=question, context=context)
print(f"\nQA  ---->")
print(f"Context: {context}")
print(f"Question: {question}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.3f}")
print(f"Start/End: {result['start']}-{result['end']}")
print(f"="*50)

# -----------------------------------------------------
# پیاده‌سازی دقیق‌تر QA
# -----------------------------------------------------

# بارگذاری مدل و توکنایزر برای QA
qa_tokenizer = AutoTokenizer.from_pretrained(MODELS['qa'])
qa_model = AutoModelForQuestionAnswering.from_pretrained(MODELS['qa'])
qa_model.to(device)

def advanced_qa(question, context):
    inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = qa_model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
    answer = qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    confidence = (torch.softmax(outputs.start_logits, dim=1)[0][answer_start] * 
                 torch.softmax(outputs.end_logits, dim=1)[0][answer_end-1]).item()
    
    return {
        "answer": answer,
        "confidence": confidence,
        "start": answer_start.item(),
        "end": answer_end.item()
    }

# تست QA پیشرفته
context = "The iPhone is a line of smartphones designed and marketed by Apple Inc."
question = "Who markets the iPhone?"

result = advanced_qa(question, context)
print(f"\nAdvanced QA ---->")
print(f"Context: {context}")
print(f"Question: {question}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"="*50)

# =====================================================
# 3.  Text Generation
# =====================================================

# -----------------------------------------------------
#  استفاده از Text Generation Pipeline
# -----------------------------------------------------

# ایجاد text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=MODELS['generation'],
    device=0 if torch.cuda.is_available() else -1
)

# تولید متن
prompt = "The future of artificial intelligence"
generated_text = text_generator(
    prompt,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True
)

print(f"\nText Generation ---->")
print(f"Prompt: {prompt}")
print(f"Generated text: {generated_text[0]['generated_text']}")
print(f"="*50)

# -----------------------------------------------------
#  کنترل شده‌تر Text Generation
# -----------------------------------------------------

# بارگذاری مدل و توکنایزر برای text generation
gen_tokenizer = AutoTokenizer.from_pretrained(MODELS['generation'])
gen_model = AutoModelForCausalLM.from_pretrained(MODELS['generation'])
gen_model.to(device)

def controlled_generation(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
    inputs = gen_tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = gen_model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=gen_tokenizer.eos_token_id
        )
    
    generated_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# تست controlled generation
prompt = "Renewable energy is important because"
generated = controlled_generation(
    prompt,
    max_length=150,
    temperature=0.8,
    top_k=30,
    top_p=0.95
)

print(f"\n Controlled Text Generation ---->")
print(f"Prompt: {prompt}")
print(f"Generated text: {generated}")
print(f"="*50)