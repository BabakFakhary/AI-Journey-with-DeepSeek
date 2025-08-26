#                                                                   به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import json
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------------------
# تحلیل احساسات با قابلیت تشخیص جملات خنثی
# ---------------------------------------------------------------------------------------------------

def fa(text):
    return get_display(arabic_reshaper.reshape(text)) 

# --------------------------------------------------------------------
# 1. تولید داده‌های واقعی فارسی (نظرات محصولات) با کلاس خنثی
# --------------------------------------------------------------------
# داده‌های نمونه از نظرات واقعی فارسی
reviews_data = [
    # داده‌های مثبت (افزوده شده)
    {"text": "عالی بود واقعا راضی هستم", "label": 1},
    {"text": "کیفیت خیلی خوبی دارد", "label": 1},
    {"text": "خرید خوبی بود توصیه می کنم", "label": 1},
    {"text": "عملکرد فوق العاده ای دارد", "label": 1},
    {"text": "ارزش خرید دارد", "label": 1},    
    {"text": "عالی بود", "label": 1},   
    {"text": "خوب بود", "label": 1},   
    {"text": "عالی", "label": 1}, 
    {"text": "خوب", "label": 1}, 
    {"text": "این محصول واقعا عالی است و کیفیت فوق العاده ای دارد", "label": 1},  # مثبت
    {"text": "قیمت مناسبی دارد و ارزش خرید دارد", "label": 1},  # مثبت
    {"text": "سریع رسید و بسته بندی عالی بود", "label": 1},  # مثبت
    {"text": "کیفیت ساخت عالی، کاملا راضی هستم", "label": 1},  # مثبت
    {"text": "کارایی فوق العاده، پیشنهاد می کنم", "label": 1},  # مثبت
    {"text": "طراحی زیبا و عملکرد عالی", "label": 1},  # مثبت
    {"text": "نسبت به قیمتش عالیه", "label": 1},  # مثبت
    {"text": "همه ویژگی گفته شده رو داره", "label": 1},  # مثبت
    # داده‌های منفی (افزوده شده)
    {"text": "خیلی بد بود ناراضی هستم", "label": 0},
    {"text": "کیفیت بسیار پایینی دارد", "label": 0},
    {"text": "پشیمان شدم از خرید", "label": 0},
    {"text": "عملکرد ضعیفی دارد", "label": 0},
    {"text": "ارزش خرید ندارد", "label": 0},   
    {"text": "خیلی بد و بی کیفیت بود، پشیمان شدم", "label": 0},  # منفی    
    {"text": "اصلا توصیه نمی کنم، waste of money", "label": 0},  # منفی    
    {"text": "محصول معیوب رسید، بسیار ناراحتم", "label": 0},  # منفی    
    {"text": "بدترین خرید عمرم بود", "label": 0},  # منفی    
    {"text": "پس از دو روز خراب شد", "label": 0},  # منفی    
    {"text": "اصلا به درد نمی خوره", "label": 0},  # منفی    
    {"text": "حیف پولم که خرج این چیز بدم کردم", "label": 0},  # منفی   
    {"text": "بد هست", "label": 0},  # منفی    
    {"text": "بد خیلی", "label": 0},  # منفی  
    # اضافه کردن داده‌های خنثی
    {"text": "این محصول معمولی است", "label": 2},  # خنثی
    {"text": "نه خوبه نه بد", "label": 2},  # خنثی
    {"text": "نه خوب نه بد", "label": 2},  # خنثی
    {"text": "محصول قابل قبولی است", "label": 2},  # خنثی
    {"text": "چیزی برای گفتن ندارم", "label": 2},  # خنثی
    {"text": "معمولی مثل بقیه محصولات", "label": 2},  # خنثی
    {"text": "نه عالی نه بد", "label": 2},  # خنثی
    {"text": "مناسب قیمتش بود", "label": 2},  # خنثی
    {"text": "قابل قبول", "label": 2},  # خنثی
    {"text": "معمولی هست", "label": 2},  # خنثی
]

# ایجاد دیتافرام
df = pd.DataFrame(reviews_data)
print(fa("داده‌های نمونه:"))
print(df.head())

# --------------------------------------------------------------------
# 2. پیش‌پردازش متن فارسی
# --------------------------------------------------------------------
def preprocess_persian_text(text):
    """پیش‌پردازش متن فارسی"""
    # حذف اعداد و علائم خاص
    text = re.sub(r'[۰-۹0-9]', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # حذف فاصله‌های اضافی
    text = re.sub(r'\s+', ' ', text)
    
    # تبدیل به حروف کوچک
    text = text.lower().strip()
    
    return text

# اعمال پیش‌پردازش
df['cleaned_text'] = df['text'].apply(preprocess_persian_text)
print(fa("\nمتن‌های پیش‌ پردازش شده:"))
print(df[['text', 'cleaned_text']].head())

# --------------------------------------------------------------------
# 3. ایجاد Vocabulary و Tokenizer ساده
# --------------------------------------------------------------------
def build_vocabulary(texts, vocab_size=1000):
    """ساخت vocabulary از متون فارسی"""
    word_counts = {}
    
    for text in texts:
        words = text.split()
        for word in words:
            if len(word) > 2:  # فقط کلمات با طول بیشتر از 2
                word_counts[word] = word_counts.get(word, 0) + 1
    
    # مرتب‌سازی بر اساس تکرار
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # ایجاد vocabulary
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (word, count) in enumerate(sorted_words[:vocab_size-2]):
        vocab[word] = i + 2
    
    return vocab

def text_to_sequence(text, vocab, max_length=20):
    """تبدیل متن به دنباله اعداد"""
    words = text.split()
    sequence = []
    
    for word in words:
        sequence.append(vocab.get(word, vocab['<UNK>']))
    
    # padding یا truncate
    if len(sequence) < max_length:
        sequence = sequence + [vocab['<PAD>']] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    
    return sequence

# ساخت vocabulary
vocab = build_vocabulary(df['cleaned_text'].tolist(), vocab_size=100)
print(fa(f"\nاندازه vocabulary: {len(vocab)}"))
print(fa("نمونه‌ای از vocabulary:"), dict(list(vocab.items())[:10]))

# تبدیل متون به sequences
max_length = 15
X_sequences = []
for text in df['cleaned_text']:
    seq = text_to_sequence(text, vocab, max_length)
    X_sequences.append(seq)

X = np.array(X_sequences)
y = np.array(df['label'])

print(fa(f"\nشکل داده‌ها: X={X.shape}, y={y.shape}"))

# --------------------------------------------------------------------
# 4. تعریف MultiHeadAttention ساده‌شده
# --------------------------------------------------------------------
class SimpleMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(SimpleMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Attention ساده‌شده
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(output)

# --------------------------------------------------------------------
# 5. تعریف Transformer Block کامل
# --------------------------------------------------------------------  
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.mha = SimpleMultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training):
        # Self-Attention
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

# --------------------------------------------------------------------
# 6. ساخت مدل کامل برای تحلیل احساسات (سه کلاسه)
# --------------------------------------------------------------------
def build_sentiment_model(vocab_size, max_length, d_model=64, num_heads=4, dff=128, num_classes=3):
    """ساخت مدل تحلیل احساسات با Transformer"""
    
    inputs = tf.keras.Input(shape=(max_length,))
    
    # Embedding Layer
    embedding = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    
    # Positional Encoding ساده‌شده
    positions = tf.range(start=0, limit=max_length, delta=1)
    positions = tf.expand_dims(positions, 0)
    positional_encoding = tf.keras.layers.Embedding(max_length, d_model)(positions)
    
    x = embedding + positional_encoding
    
    # Transformer Block
    transformer_block = TransformerBlock(d_model, num_heads, dff)
    x = transformer_block(x, training=True)
    
    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Classification Head (اکنون سه کلاسه)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # تغییر به softmax نسبت به کد SentimentAnalysisWithTransformer.py
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
# def build_sentiment_model(vocab_size, max_length, d_model=32, num_heads=2, dff=64, num_classes=3):
#     """ساخت مدل ساده‌تر"""
    
#     inputs = tf.keras.Input(shape=(max_length,))
    
#     # Embedding
#     embedding = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    
#     # جایگزینی Transformer با LSTM ساده‌تر
#     x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(embedding)
#     x = tf.keras.layers.Dropout(0.3)(x)
    
#     # Classification Head
#     outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# ساخت مدل با سه کلاس
vocab_size = len(vocab)
num_classes = 3  # منفی، مثبت، خنثی
model = build_sentiment_model(vocab_size, max_length, num_classes=num_classes)

print(fa("\nخلاصه مدل:"))
model.summary()

# کامپایل مدل (اکنون با loss مناسب برای چندکلاسه)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # تغییر به این تابع زیان
    metrics=['accuracy']
)

# --------------------------------------------------------------------
# 7. آموزش مدل
# --------------------------------------------------------------------
print(fa("\n🔨 آموزش مدل..."))

# تقسیم داده به train و test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. اضافه کردن وزن کلاس برای مقابله با عدم تعادل
class_weights = {0: 2.0, 1: 2.0, 2: 1.0}  # وزن بیشتر برای کلاس‌های اقلیت

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=4,  # بچ سایز کوچکتر
    validation_split=0.2,
    class_weight=class_weights,  # اضافه کردن وزن کلاس
    verbose=1
)

# --------------------------------------------------------------------
# 8. ارزیابی مدل
# --------------------------------------------------------------------
print(fa("\n📊 ارزیابی مدل..."))

# پیش‌بینی روی داده تست
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # گرفتن کلاس با بیشترین احتمال

accuracy = accuracy_score(y_test, y_pred)
print(fa(f"دقت مدل: {accuracy:.2f}"))

print(fa("\nگزارش طبقه‌بندی:"))
# تعریف نام کلاس‌ها برای گزارش
class_names = [fa('منفی'), fa('مثبت'), fa('خنثی')]
print(classification_report(y_test, y_pred, target_names=class_names))

# --------------------------------------------------------------------
# 9. تست مدل روی جملات جدید
# --------------------------------------------------------------------
print(fa("\n🧪 تست مدل روی جملات جدید:"))

test_sentences = [
    "این محصول عالی است",  # مثبت
    "خیلی بد بود",         # منفی
    "کیفیت خوبی دارد",     # مثبت
    "پشیمان شدم",          # منفی
    "محصول معمولی است",    # خنثی
    "نه خوبه نه بد",       # خنثی
    "قابل قبول بود",       # خنثی
]

# تابع پیش‌بینی جدید برای سه کلاس
def predict_sentiment(text, model, vocab, max_length=15):
    # پیش‌پردازش
    cleaned = preprocess_persian_text(text)
    sequence = text_to_sequence(cleaned, vocab, max_length)
    sequence = np.array([sequence])
    
    # پیش‌بینی
    prediction_probs = model.predict(sequence)[0]
    predicted_class = np.argmax(prediction_probs)
    confidence = prediction_probs[predicted_class]
    
    # نگاشت کلاس به نام احساس
    sentiment_map = {
        0: "منفی 👎",
        1: "مثبت 👍", 
        2: "خنثی 😐"
    }
    
    return sentiment_map[predicted_class], confidence, prediction_probs

for sentence in test_sentences:
    # پیش‌بینی
    sentiment, confidence, probs = predict_sentiment(sentence, model, vocab, max_length)
    
    print(fa(f"جمله: '{sentence}'"))
    print(fa(f"احساس: {sentiment} (اعتماد: {confidence:.2f})"))
    # استفاده از f-string برای حل مشکل
    print(fa(f"توزیع احتمالات: منفی={probs[0]:.2f}, مثبت={probs[1]:.2f}, خنثی={probs[2]:.2f}"))
    print("-" * 50)

# --------------------------------------------------------------------
# 10. تجسم نتایج
# --------------------------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label=fa('دقت آموزش'))
plt.plot(history.history['val_accuracy'], label=fa('دقت validation'))
plt.title(fa('دقت مدل'))
plt.xlabel(fa('دوره'))
plt.ylabel(fa('دقت'))
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label=fa('خطای آموزش'))
plt.plot(history.history['val_loss'], label=fa('خطای validation'))
plt.title(fa('خطای مدل'))
plt.xlabel(fa('دوره'))
plt.ylabel(fa('خطا'))
plt.legend()

plt.tight_layout()
plt.show()

print(fa("\n🎉 مدل Transformer برای تحلیل احساسات فارسی (سه کلاسه) آماده است!"))