#                                                                     به نام خدا   
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install transformers datasets accelerate flask
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
from flask import Flask, request, jsonify
import threading
import requests
import time 
import json


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

# =====================================================
# 4.  مثال عملی: سیستم کامل NLP
# =====================================================

# -----------------------------------------------------
#  ایجاد سیستم NLP چندمنظوره
# -----------------------------------------------------

class AdvancedNLPSystem:
    def __init__(self):
        # بارگذاری همه مدل‌ها
        self.models = {}
        self.tokenizers = {}
        
        # بارگذاری مدل‌های مختلف
        self._load_models()
    
    def _load_models(self):
        """بارگذاری همه مدل‌های لازم"""
        print("Loading models...")
        
        # مدل برای classification
        self.models['classification'] = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=2
        )
        self.tokenizers['classification'] = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # مدل برای NER
        self.models['ner'] = AutoModelForTokenClassification.from_pretrained(MODELS['ner'])
        self.tokenizers['ner'] = AutoTokenizer.from_pretrained(MODELS['ner'])
        
        # مدل برای QA
        self.models['qa'] = AutoModelForQuestionAnswering.from_pretrained(MODELS['qa'])
        self.tokenizers['qa'] = AutoTokenizer.from_pretrained(MODELS['qa'])
        
        # انتقال به GPU
        for model in self.models.values():
            model.to(device)
        
        print("All models loaded successfully!")
    
    def analyze_text(self, text):
        """آنالیز کامل متن"""
        results = {}
        
        # Sentiment Analysis
        results['sentiment'] = self._analyze_sentiment(text)
        
        # Named Entity Recognition
        results['entities'] = self._extract_entities(text)
        
        # Text Summary (simulated)
        results['summary'] = self._generate_summary(text)
        
        return results
    
    def _analyze_sentiment(self, text):
        """تحلیل احساسات"""
        inputs = self.tokenizers['classification'](text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.models['classification'](**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1)
        sentiment = "Positive" if torch.argmax(probs) == 1 else "Negative"
        
        return {
            'sentiment': sentiment,
            'confidence': probs[0][torch.argmax(probs)].item(),
            'probabilities': probs.cpu().numpy()[0].tolist()
        }
    
    def _extract_entities(self, text):
        """استخراج موجودیت‌های نامدار"""
        ner_pipeline = pipeline(
            "token-classification",
            model=self.models['ner'],
            tokenizer=self.tokenizers['ner'],
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        
        return ner_pipeline(text)
    
    def _generate_summary(self, text):
        """ایجاد خلاصه متن (ساده)"""
        # در واقعیت از مدل summarization استفاده می‌شود
        sentences = text.split('.')
        if len(sentences) > 3:
            return '.'.join(sentences[:2]) + '.'
        return text

# ایجاد و تست سیستم
nlp_system = AdvancedNLPSystem()

# متن نمونه برای آنالیز
sample_text = """
Apple Inc. is planning to open a new research facility in Tokyo, Japan next year. 
The company announced this exciting news during their quarterly earnings call. 
This move demonstrates Apple's commitment to expanding its global presence and 
investing in artificial intelligence research. The new facility will create 
approximately 500 jobs for local engineers and researchers.
"""

print("Analyzing text...")
analysis = nlp_system.analyze_text(sample_text)

print("\n=== TEXT ANALYSIS RESULTS ===")
print(f"Sentiment: {analysis['sentiment']['sentiment']} (confidence: {analysis['sentiment']['confidence']:.3f})")

print("\nNamed Entities:")
for entity in analysis['entities']:
    print(f"  {entity['word']:20} -> {entity['entity_group']} (score: {entity['score']:.3f})")

print(f"\nSummary: {analysis['summary']}")

test_text = """
  Apple Inc. is an American multinational technology company that revolutionized the technology sector through its innovation of computer software, personal computers, mobile tablets, smartphones, and computer peripherals.
  One of the most recognizable brands in the world, Apple created the first commercially successful personal computer and was also the first to bring the graphical user interface (GUI) into mass adoption.
  Founded by Steve Jobs and Steve Wozniak in 1976, Apple set new benchmarks in product innovation, user-centric functionality, aesthetics and design, and multiproduct integration. Apple redefined and transformed the capabilities of modern computing. Further, Apple innovated the industry by establishing a marketplace ecosystem for third-party application developers, leveraging this new economy to expand its products’ functionalities and strengthen its position. The company is headquartered in Cupertino, California.
"""

print("Analyzing test text...")
print(f"="*50)
analysis = nlp_system.analyze_text(test_text)

print("\n=== TEXT ANALYSIS RESULTS ===")
print(f"Sentiment: {analysis['sentiment']['sentiment']} (confidence: {analysis['sentiment']['confidence']:.3f})")

print("\nNamed Entities:")
for entity in analysis['entities']:
    print(f"  {entity['word']:20} -> {entity['entity_group']} (score: {entity['score']:.3f})")

print(f"\nSummary: {analysis['summary']}")

# -----------------------------------------------------
#  Define API
#   ایجاد ای پی ای ساده برای تست کد نوشته شده در 
#     ایجاد سیستم NLP چندمنظوره   
# -----------------------------------------------------

class NLPAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.nlp_system = AdvancedNLPSystem()
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.route('/analyze', methods=['POST'])
        def analyze_text():
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
            try:
                results = self.nlp_system.analyze_text(text)
                # تبدیل float32 به float
                sanitized_results = self._sanitize_results(results)
                return jsonify(sanitized_results)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy', 'device': str(device)})
    
    def _sanitize_results(self, results):
        """تبدیل انواع غیرقابل سریال‌سازی به انواع استاندارد"""
        
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif hasattr(obj, 'tolist'):  # برای tensorها و arrayها
                return obj.tolist()
            else:
                return obj
        
        return convert(results)
    
    def run(self, host='0.0.0.0', port=5000):
        print(f"="*50)
        print(f"Starting NLP API on {host}:{port}")
        self.app.run(host=host, port=port, debug=False)

# اجرای API در background
def start_api():
    api = NLPAPI()
    api.run()

def test_api():
    # داده برای ارسال
    data = {
        "text": "Apple is opening a new store in San Francisco next month"
    }
    
    try:
        # ارسال درخواست POST
        response = requests.post('http://localhost:5000/analyze', json=data)
        
        # دریافت نتایج
        print("Test Api...")
        results = response.json()
        print(results)
    except Exception as e:
        print(f"API test failed: {e}")
        print("Make sure the API is running on port 5000")

# اگر بخواهید API را اجرا کنید:
# threading.Thread(target=start_api, daemon=True).start()
# print("API is running in background...")

# ********************************************************************
#                            Run And Test API 
# ********************************************************************
# اگر بخواهید API را اجرا کنید، این خطوط را آنکامنت کنید:
if __name__ == '__main__':
    # انتخاب کنید کدام بخش را می‌خواهید اجرا کنید:
    
    # گزینه ۱: فقط API را اجرا کن
    print(f"="*50)
    print(f"*"*50)
    print("Starting NLP API Server...")
    start_api()
    
    # گزینه ۲: API در background و سپس تست
    print(f"*"*50)
    print("Starting API in background...")
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    # 
    print(f"*"*50)
    print("Waiting for API to start...")
    time.sleep(5)  # منتظر بمان API بالا بیاید
    # 
    print(f"-"*50)
    print("Testing API...")
    test_api()
    # 
    # # منتظر بمان تا کاربر Enter بزند
    input("Press Enter to stop...")



