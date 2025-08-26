#                                                                  به نام خدا
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
# تحلیل احساسات
# ---------------------------------------------------------------------------------------------------

def fa(text):
    return get_display(arabic_reshaper.reshape(text)) 

# --------------------------------------------------------------------
# 1. تولید داده‌های واقعی فارسی (نظرات محصولات)
# --------------------------------------------------------------------
# داده‌های نمونه از نظرات واقعی فارسی
reviews_data = [
    {"text": "این محصول واقعا عالی است و کیفیت فوق العاده ای دارد", "label": 1},
    {"text": "خیلی بد و بی کیفیت بود، پشیمان شدم", "label": 0},
    {"text": "قیمت مناسبی دارد و ارزش خرید دارد", "label": 1},
    {"text": "اصلا توصیه نمی کنم، waste of money", "label": 0},
    {"text": "سریع رسید و بسته بندی عالی بود", "label": 1},
    {"text": "محصول معیوب رسید، بسیار ناراحتم", "label": 0},
    {"text": "کیفیت ساخت عالی، کاملا راضی هستم", "label": 1},
    {"text": "بدترین خرید عمرم بود", "label": 0},
    {"text": "کارایی فوق العاده، پیشنهاد می کنم", "label": 1},
    {"text": "پس از دو روز خراب شد", "label": 0},
    {"text": "طراحی زیبا و عملکرد عالی", "label": 1},
    {"text": "اصلا به درد نمی خوره", "label": 0},
    {"text": "نسبت به قیمتش عالیه", "label": 1},
    {"text": "حیف پولم که خرج این چیز بدم کردم", "label": 0},
    {"text": "همه ویژگی های承诺 شده رو داره", "label": 1},
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
  #  Self-Attention (توجه به خود) : کاری که می‌کند: به هر کلمه می‌گوید: "به همه کلمات دیگر نگاه کن و ببین چقدر به هرکدام باید توجه کنی
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
    # Transformer Block: 
      # روابط بین همه کلمات را پیدا می‌کند
      # اطلاعات را ترکیب می‌کند 
      # پردازش عمیق انجام می‌دهد
      # اطلاعات را غنی‌تر می‌کند 
      # حفظ اطلاعات اصلی + اضافه کردن اطلاعات جدید
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        #  Self-Attention (توجه به خود) : کاری که می‌کند: به هر کلمه می‌گوید: "به همه کلمات دیگر نگاه کن و ببین چقدر به هرکدام باید توجه کنی
        self.mha = SimpleMultiHeadAttention(d_model, num_heads)

        # Feed Forward Network (شبکه پیش‌خور) : کاری که می‌کند: "یک پردازش اضافی و غیرخطی روی اطلاعات انجام بده"
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        # Residual Connection و Layer Norm  : کاری که می‌کند: "باز هم اطلاعات را ترکیب و استاندارد کن"
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training):
        # Self-Attention
        attn_output = self.mha(x, x, x)  # Q, K, V همه x هستند (Self-Attention)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual connection
        
        # Feed Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection
        
        return out2

# --------------------------------------------------------------------
# 6. ساخت مدل کامل برای تحلیل احساسات
# --------------------------------------------------------------------
def build_sentiment_model(vocab_size, max_length, d_model=64, num_heads=4, dff=128):
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
    
    # Classification Head
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# ساخت مدل
vocab_size = len(vocab)
model = build_sentiment_model(vocab_size, max_length)

print(fa("\nخلاصه مدل:"))
model.summary()

# کامپایل مدل
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------------------------
# 7. آموزش مدل
# --------------------------------------------------------------------
print(fa("\n🔨 آموزش مدل..."))

# تقسیم داده به train و test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# --------------------------------------------------------------------
# 8. ارزیابی مدل
# --------------------------------------------------------------------
print(fa("\n📊 ارزیابی مدل..."))

# پیش‌بینی روی داده تست
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_binary)
print(fa(f"دقت مدل: {accuracy:.2f}"))

print(fa("\nگزارش طبقه‌بندی:"))
print(classification_report(y_test, y_pred_binary))

# --------------------------------------------------------------------
# 9. تست مدل روی جملات جدید
# --------------------------------------------------------------------
print(fa("\n🧪 تست مدل روی جملات جدید:"))

test_sentences = [
    "این محصول عالی است",  # مثبت
    "خیلی بد بود",         # منفی
    "کیفیت خوبی دارد",     # مثبت
    "پشیمان شدم",          # منفی
]

for sentence in test_sentences:
    # پیش‌پردازش
    cleaned = preprocess_persian_text(sentence)
    sequence = text_to_sequence(cleaned, vocab, max_length)
    sequence = np.array([sequence])
    
    # پیش‌بینی
    prediction = model.predict(sequence)[0][0]
    sentiment = "مثبت 👍" if prediction > 0.5 else "منفی 👎"
    
    print(fa(f"جمله: '{sentence}'"))
    print(fa(f"احساس: {sentiment} (اعتماد: {prediction:.2f})"))
    print("-" * 40)

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

print(fa("\n🎉 مدل Transformer برای تحلیل احساسات فارسی آماده است!"))