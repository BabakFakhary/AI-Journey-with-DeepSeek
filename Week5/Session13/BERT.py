#                                                                    به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install transformers datasets
# pip install transformers[torch]
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# ============================================================================================
#                                       BERT  مثال عملی از  
# ============================================================================================


# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

# ---------------------------------------------------
# بررسی وجود GPU
# ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=" * 50)
print(f"Using device: {device}")
print(f"=" * 50)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ---------------------------------------------------
# ایجاد دیتاست متن برای طبقه‌بندی احساسات
# ---------------------------------------------------
texts = [
    "I absolutely love this movie! The acting was fantastic.",
    "This film is terrible. Waste of time and money.",
    "Wonderful storyline with amazing character development.",
    "Poor direction and weak script made it boring.",
    "The cinematography is stunning and the music is perfect.",
    "Horrible acting, I couldn't even finish watching it.",
    "A masterpiece of modern cinema, highly recommended!",
    "Disappointing ending ruined the whole movie for me.",
    "Brilliant performance by all actors involved.",
    "The plot was predictable and the dialogue was cheesy.",
    "Outstanding visual effects and gripping narrative.",
    "Boring and unoriginal, nothing new to see here.",
    "Exceptional directing and powerful performances.",
    "Weak character development and confusing plot.",
    "Emotionally moving and beautifully crafted.",
    "Poorly executed with many plot holes.",
    "Captivating from start to finish, a true gem.",
    "Mediocre at best, failed to impress.",
    "Innovative storytelling with great depth.",
    "Lackluster and forgettable experience."
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative

# ---------------------------------------------------
# ایجاد DataFrame
# ---------------------------------------------------
df = pd.DataFrame({'text': texts, 'label': labels})
print("Dataset:")
print(df.head())

# ---------------------------------------------------
# تقسیم داده به train/validation/test
# ---------------------------------------------------
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")
print(f"Class distribution in train: {train_df['label'].value_counts().to_dict()}")

# ---------------------------------------------------
# Dataset format
# ---------------------------------------------------
# تبدیل به Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

print("Dataset structure:")
print(train_dataset)

# ---------------------------------------------------
# بارگذاری توکنایزر BERT
# ---------------------------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# توکنایز کردن داده‌ها
def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=128
    )

# اعمال توکنایزینگ
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

print("Tokenized example:")
print(tokenized_train[0])

# ---------------------------------------------------
# بارگذاری مدل BERT
# ---------------------------------------------------
# بارگذاری مدل BERT برای طبقه‌بندی
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2, # بعلت نوع خروجی   0 یا 1
    output_attentions=False,
    output_hidden_states=False
)

# انتقال مدل به GPU اگر available باشد
model.to(device)

print("Model loaded successfully!")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------
# تنظیمات آموزش
# ---------------------------------------------------
# تعریف metric برای evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# تنظیمات آموزش
training_args = TrainingArguments(
    output_dir='./bert-sentiment-results',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy ='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    seed=42
)


# ---------------------------------------------------
# Fine-tuning مدل
# ---------------------------------------------------

# ایجاد Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# شروع آموزش
print("Starting fine-tuning...")
train_results = trainer.train()

# ذخیره مدل
trainer.save_model()
tokenizer.save_pretrained('./bert-sentiment-model')

print("Fine-tuning completed!")
print(f"Training results: {train_results}")


# ---------------------------------------------------
# ارزیابی مدل Fine-tuned
# ---------------------------------------------------

# ارزیابی روی validation set
eval_results = trainer.evaluate()
print("Validation results:")
print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

# پیش‌بینی روی validation set
val_predictions = trainer.predict(tokenized_val)
val_preds = np.argmax(val_predictions.predictions, axis=1)

print("\nValidation Classification Report:")
print(classification_report(val_df['label'], val_preds, target_names=['Negative', 'Positive']))


# ارزیابی روی test set
test_predictions = trainer.predict(tokenized_test)
test_preds = np.argmax(test_predictions.predictions, axis=1)

print("Test results:")
print(f"Test Loss: {test_predictions.metrics['test_loss']:.4f}")
print(f"Test Accuracy: {test_predictions.metrics['test_accuracy']:.4f}")

print("\nTest Classification Report:")
print(classification_report(test_df['label'], test_preds, target_names=['Negative', 'Positive']))


# ---------------------------------------------------
#  تحلیل پیش‌بینی‌ها
# ---------------------------------------------------

# تحلیل پیش‌بینی‌ها
test_df['prediction'] = test_preds
test_df['confidence'] = np.max(torch.softmax(torch.tensor(test_predictions.predictions), dim=1).numpy(), axis=1) # اعتماد به نفس

print("Sample predictions:")
for i, row in test_df.iterrows():
    print(f"\nText: {row['text']}")
    print(f"True: {'Positive' if row['label'] == 1 else 'Negative'}")
    print(f"Pred: {'Positive' if row['prediction'] == 1 else 'Negative'}")
    print(f"Confidence: {row['confidence']:.3f}")


# --------------------------------------------------------------------------
#                             طبقه‌بندی احساسات 
# --------------------------------------------------------------------------

# ---------------------------------------------------
# استفاده از مدل برای پیش‌بینی جدید
# ---------------------------------------------------

# تابع پیش‌بینی برای متن جدید
def predict_sentiment(text, model, tokenizer):
    # توکنایز کردن متن
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=128
    )
    
    # انتقال به device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # پیش‌بینی
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # محاسبه احتمال‌ها
    probs = torch.softmax(outputs.logits, dim=1)
    pred_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_label].item()
    
    return pred_label, confidence

# تست با جملات جدید
test_sentences = [
    "This movie is absolutely fantastic and wonderful!",
    "I hated every minute of this terrible film.",
    "The acting was mediocre but the story was good.",
    "A complete waste of time, nothing worked in this movie."
]

print(f"\n")
print(f"=" * 50)
print("Predictions for new sentences:")
print(f"=" * 50)
for sentence in test_sentences:
    label, confidence = predict_sentiment(sentence, model, tokenizer)
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"\nText: {sentence}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.3f})")


# ---------------------------------------------------
# تجسم توجه (Attention)‌
# ---------------------------------------------------

# گرفتن attention weights (برای تجسم)
def get_attention(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    return outputs.attentions

# تجسم attention برای یک جمله
sample_text = "I love this amazing movie but the ending was disappointing"
attentions = get_attention(sample_text, model, tokenizer)

print(f"Number of attention layers: {len(attentions)}")
print(f"Attention shape for first layer: {attentions[0].shape}")

# توجه: برای تجسم کامل attention maps به کتابخانه‌های اضافی نیاز داریم


# --------------------------------------------------------------------------
#                             ذخیره و استفاده از مدل 
# --------------------------------------------------------------------------

# ---------------------------------------------------
# ذخیره مدل نهایی
# ---------------------------------------------------

# ذخیره مدل و توکنایزر
model.save_pretrained('./final-bert-sentiment-model')
tokenizer.save_pretrained('./final-bert-sentiment-model')

print("Model and tokenizer saved successfully!")

# ذخیره config
import json

model_config = {
    'model_name': 'bert-base-uncased',
    'num_labels': 2,
    'max_length': 128,
    'labels': {'0': 'Negative', '1': 'Positive'}
}

with open('./final-bert-sentiment-model/config.json', 'w') as f:
    json.dump(model_config, f, indent=2)


# ---------------------------------------------------
# بارگذاری مدل ذخیره شده
# ---------------------------------------------------

# بارگذاری مدل ذخیره شده
loaded_model = BertForSequenceClassification.from_pretrained('./final-bert-sentiment-model')
loaded_tokenizer = BertTokenizer.from_pretrained('./final-bert-sentiment-model')

loaded_model.to(device)

print("Model loaded successfully from disk!")

# تست مدل بارگذاری شده
test_text = "This is an excellent film with great acting"
label, confidence = predict_sentiment(test_text, loaded_model, loaded_tokenizer)
sentiment = "Positive" if label == 1 else "Negative"

print(f"Text: {test_text}")
print(f"Prediction: {sentiment} (Confidence: {confidence:.3f})")



# --------------------------------------------------------------------------
#                            ایجاد API ساده
# --------------------------------------------------------------------------

# کلاس برای inference آسان
class SentimentAnalyzer:
    def __init__(self, model_path):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def analyze(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()
        
        return {
            'sentiment': 'Positive' if pred_label == 1 else 'Negative',
            'confidence': confidence,
            'probabilities': probs.cpu().numpy()[0].tolist()
        }

# استفاده از analyzer
analyzer = SentimentAnalyzer('./final-bert-sentiment-model')

texts_to_analyze = [
    "I absolutely love this movie!",
    "This was terrible and boring.",
    "It was okay, nothing special."
]

print("Sentiment Analysis Results:")
for text in texts_to_analyze:
    result = analyzer.analyze(text)
    print(f"\nText: {text}")
    print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
    print(f"Probabilities: {result['probabilities']}")
