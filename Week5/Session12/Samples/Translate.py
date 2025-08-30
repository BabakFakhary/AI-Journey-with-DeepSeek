#                                                                   به نام خدا
#                                                                   به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def fa(text):
    return get_display(arabic_reshaper.reshape(text)) 

print(fa("🐍 کتابخانه‌ها وارد شدند"))

# -------------------------------------------------------------------------------------------------
#  یک دیتاست کوچک از جملات انگلیسی و معادل فارسی آنها ایجاد می‌کنیم
#  این دیتاست برای نمایش مفهوم ترجمه با توجه کافی است 
# مدل فعلی فقط یک کلمه ترجمه می‌کند
# -------------------------------------------------------------------------------------------------

# =============================================================================
# 1. ایجاد دیتاست نمونه (انگلیسی به فارسی)
# =============================================================================
print("\n" + "="*60)
print(fa("1. ایجاد دیتاست نمونه (انگلیسی به فارسی)"))
print("="*60)

# دیتاست کوچک برای نمایش مفهوم
english_sentences = [
    "I love machine learning",
    "Attention is important",
    "Transformers are powerful",
    "Neural networks learn",
    "Deep learning is amazing",
    "AI is the future",
    "Natural language processing",
    "Computer vision applications",
    "Hello world program",
    "GPT models are large"
]

persian_sentences = [
    "من عاشق یادگیری ماشین هستم",
    "توجه مهم است", 
    "ترانسفورمرها قدرتمند هستند",
    "شبکه های عصبی یاد می گیرند",
    "یادگیری عمیق شگفت انگیز است",
    "هوش مصنوعی آینده است",
    "پردازش زبان طبیعی",
    "کاربردهای بینایی کامپیوتر",
    "برنامه سلام دنیا",
    "مدل های جی پی تی بزرگ هستند"
]

print(fa("📊 دیتاست ایجاد شد:"))
for i, (eng, per) in enumerate(zip(english_sentences, persian_sentences)):
    print(f"{i+1:2d}. EN: {eng:<30} FA: {fa(per)}")

# =============================================================================
# 2. پیش‌پردازش متن و Tokenization
  # از Tokenizer keras برای تبدیل کلمات به اعداد استفاده می‌کنیم
  # oov_token="<OOV>" برای مدیریت کلمات ناشناخته اضافه می‌شود
  # word_index دیکشنری mapping کلمات به اعداد است
# =============================================================================
print("\n" + "="*60)
print(fa("2. پیش‌پردازش متن و Tokenization"))
print("="*60)

# توکنایزر برای انگلیسی
  # یافتن کلمات موجود در متن ورودی
eng_tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',  # فیلتر کاراکترهای خاص
    oov_token="<OOV>"  # توکن برای کلمات ناشناخته
)
eng_tokenizer.fit_on_texts(english_sentences)
eng_vocab_size = len(eng_tokenizer.word_index) + 1  # +1 برای padding

# توکنایزر برای فارسی
  # یافتن کلمات موجود در متن ورودی
per_tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', # فیلتر کاراکترهای خاص
    oov_token="<OOV>" # توکن برای کلمات ناشناخته
)
per_tokenizer.fit_on_texts(persian_sentences)
per_vocab_size = len(per_tokenizer.word_index) + 1

print(fa(f"📝 اندازه دایره واژگان انگلیسی: {eng_vocab_size}"))
print(fa(f"📝 اندازه دایره واژگان فارسی: {per_vocab_size}"))
print(fa(f"📋 واژگان انگلیسی: {list(eng_tokenizer.word_index.keys())[:10]}..."))
print(fa(f"📋 واژگان فارسی: {list(per_tokenizer.word_index.keys())[:10]}..."))

# تبدیل متن به دنباله اعداد
  # english_sentences = ["I love machine learning"]  
  # eng_tokenizer.word_index = { 'machine': 1, 'learning': 2, 'love': 3, 'i': 4, 'attention': 5, 'is': 6, 'important': 7, 'transformers': 8, 'are': 9, 'powerful': 10}
  # خروجی : # "I love machine learning" → [4, 3, 1, 2]
eng_sequences = eng_tokenizer.texts_to_sequences(english_sentences)
per_sequences = per_tokenizer.texts_to_sequences(persian_sentences)

print(fa(f"\n🔢 نمونه دنباله‌های انگلیسی: {eng_sequences[0]}"))
print(fa(f"🔢 نمونه دنباله‌های فارسی: {per_sequences[0]}"))

# =============================================================================
# 3. Padding برای یکسان کردن طول دنباله‌ها
# =============================================================================
print("\n" + "="*60)
print(fa("3. Padding برای یکسان کردن طول دنباله‌ها"))
print("="*60)

max_len = 10  # حداکثر طول جمله

eng_padded = tf.keras.preprocessing.sequence.pad_sequences(
    eng_sequences, maxlen=max_len, padding='post', truncating='post')

per_padded = tf.keras.preprocessing.sequence.pad_sequences(
    per_sequences, maxlen=max_len, padding='post', truncating='post')

print(fa(f"📏 طول پس از padding: {max_len}"))
print(fa(f"📊 شکل داده‌های انگلیسی: {eng_padded.shape}"))
print(fa(f"📊 شکل داده‌های فارسی: {per_padded.shape}"))

print(fa(f"\n🔤 نمونه جمله انگلیسی پس از padding: {eng_padded[0]}"))
print(fa(f"🔤 نمونه جمله فارسی پس از padding: {per_padded[0]}"))

# =============================================================================
# 4. تعریف لایه توجه چندسری (Multi-Head Attention)
# =============================================================================
print("\n" + "="*60)
print(fa("4. تعریف لایه توجه چندسری (Multi-Head Attention)"))
print("="*60)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # اطمینان از اینکه d_model بر num_heads بخش پذیر است
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads  # عمق هر سر توجه
        
        # لایه‌های Dense برای تولید Q, K, V
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        # لایه Dense نهایی
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """تقسیم آخرین بعد به (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        
        # تولید Q, K, V
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        # تقسیم به سرهای مختلف
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # محاسبه توجه
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # ادغام سرها
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        # لایه نهایی
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        """محاسبه توجه نقطه‌ای مقیاس‌شده"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        
        # مقیاس کردن
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # اعمال ماسک اگر وجود داشته باشد
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # محاسبه وزن‌های توجه
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        
        # ضرب در مقادیر
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        
        return output, attention_weights

print(fa("✅ لایه MultiHeadAttention تعریف شد"))

# =============================================================================
# 5. ساخت مدل ترجمه با مکانیزم توجه
# =============================================================================
print("\n" + "="*60)
print(fa("5. ساخت مدل ترجمه با مکانیزم توجه"))
print("="*60)

# مدل ساده‌تر و اصلاح شده
class SimpleTranslationModel(tf.keras.Model):
    def __init__(self, vocab_size_eng, vocab_size_per, d_model):
        super(SimpleTranslationModel, self).__init__()
        
        self.eng_embedding = tf.keras.layers.Embedding(vocab_size_eng, d_model)
        self.per_embedding = tf.keras.layers.Embedding(vocab_size_per, d_model)
        
        self.attention = MultiHeadAttention(d_model, 2)
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(vocab_size_per, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(0.1)
        
    def call(self, inputs):
        eng_input, per_input = inputs
        
        # Embedding
        eng_embedded = self.eng_embedding(eng_input)
        per_embedded = self.per_embedding(per_input)
        
        # توجه
        context, _ = self.attention(per_embedded, eng_embedded, eng_embedded)
        context = self.dropout(context)
        
        # پردازش نهایی
        pooled = self.global_pool(context)
        x = self.dense1(pooled)
        x = self.dropout(x)
        output = self.dense2(x)
        
        return output

# پارامترهای مدل
vocab_size_eng = eng_vocab_size
vocab_size_per = per_vocab_size
d_model = 64  # بعد embedding و hidden states

print(fa(f"🔧 پارامترهای مدل:"))
print(fa(f"   - vocab_size_eng: {vocab_size_eng}"))
print(fa(f"   - vocab_size_per: {vocab_size_per}"))
print(fa(f"   - d_model: {d_model}"))

# ایجاد مدل
model = SimpleTranslationModel(eng_vocab_size, per_vocab_size, d_model)
print(fa("✅ مدل ترجمه ایجاد شد"))

# =============================================================================
# 6. تست مدل با داده‌های نمونه
# =============================================================================
print("\n" + "="*60)
print(fa("6. تست مدل با داده‌های نمونه"))
print("="*60)

# انتخاب یک نمونه برای تست
sample_idx = 0
# این خط کد یک نمونه از داده‌های انگلیسی را انتخاب کرده و به تانسور تبدیل می‌کند تا برای مدل قابل استفاده باشد.
sample_eng = tf.constant([eng_padded[sample_idx]])  # جمله انگلیسی
sample_per = tf.constant([per_padded[sample_idx]])   # جمله فارسی

print(fa(f"🔍 تست مدل با نمونه {sample_idx + 1}:"))
print(fa(f"   EN: {english_sentences[sample_idx]}"))
print(fa(f"   FA: {persian_sentences[sample_idx]}"))
print(fa(f"   EN tokens: {sample_eng.numpy()[0]}")) # این خط کد یک تانسور را به آرایه نام پای تبدیل کرده و اولین نمونه از آرایه را برمی‌گرداند.
print(fa(f"   FA tokens: {sample_per.numpy()[0]}")) # این خط کد یک تانسور را به آرایه نام پای تبدیل کرده و اولین نمونه از آرایه را برمی‌گرداند.

# اجرای مدل
output = model([sample_eng, sample_per])

print(fa(f"\n📊 خروجی مدل:"))
print(fa(f"   Output shape: {output.shape}"))
print(fa(f"   نمونه خروجی: {output.numpy()[0, :5]}..."))  # 5 مقدار اول

# =============================================================================
# 7. کامپایل و آموزش مدل
# =============================================================================
print("\n" + "="*60)
print(fa("7. کامپایل و آموزش مدل"))
print("="*60)

# کامپایل مدل
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(fa("✅ مدل کامپایل شد"))

# تقسیم داده به آموزش و validation
X_train, X_val, y_train, y_val = train_test_split(
    eng_padded, per_padded, test_size=0.2, random_state=42
)

# تغییر شکل داده‌های هدف برای مطابقت با مدل
# مدل ما یک خروجی برای هر نمونه تولید می‌کند (not per token)
# بنابراین باید target را به شکل مناسب درآوریم

# برای مدل sequence-to-vector، باید target را به شکل (batch_size,) داشته باشیم
# ما از اولین توکن غیرصفر به عنوان target استفاده می‌کنیم
def get_first_nonzero_token(sequences):
    targets = []
    for seq in sequences:
        for token in seq:
            if token != 0:
                targets.append(token)
                break
        else:
            targets.append(0)  # اگر همه صفر بودند
    return np.array(targets)

y_train_target = get_first_nonzero_token(y_train)
y_val_target = get_first_nonzero_token(y_val)

print(fa(f"📊 داده‌های آموزش: {X_train.shape[0]} نمونه"))
print(fa(f"📊 داده‌های validation: {X_val.shape[0]} نمونه"))
print(fa(f"📊 شکل y_train_target: {y_train_target.shape}"))
print(fa(f"📊 شکل y_val_target: {y_val_target.shape}"))

# آموزش مدل
print(fa("\n🎯 شروع آموزش مدل..."))
history = model.fit(
    [X_train, y_train],  # ورودی‌های encoder و decoder
    y_train_target,      # هدف‌ها
    epochs=50,
    batch_size=4,
    validation_data=(
        [X_val, y_val],
        y_val_target
    ),
    verbose=1
)

print(fa("✅ آموزش مدل کامل شد"))

# =============================================================================
# 8. توابع کمکی برای ترجمه
# =============================================================================
print("\n" + "="*60)
print(fa("8. توابع کمکی برای ترجمه"))
print("="*60)

def simple_translate(model, sentence, eng_tokenizer, per_tokenizer, max_len=10):
    """ترجمه ساده یک جمله"""
    try:
        # پیش‌پردازش جمله ورودی
        sequence = eng_tokenizer.texts_to_sequences([sentence])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=max_len, padding='post')
        
        # استفاده از جمله خالی به عنوان ورودی decoder
        empty_decoder_input = tf.constant([[0] * max_len])
        
        # پیش‌بینی
        predictions = model([tf.constant(padded), empty_decoder_input])
        predicted_id = tf.argmax(predictions, axis=-1).numpy()[0]
        
        # تبدیل به متن
        if predicted_id > 0 and predicted_id < per_vocab_size:
            word = per_tokenizer.index_word.get(predicted_id, '')
            return word if word and word != '<OOV>' else "ترجمه ناموفق"
        
        return "ترجمه ناموفق"
    
    except Exception as e:
        return f"خطا در ترجمه: {e}"

# تست تابع ترجمه
test_sentence = "I love machine learning"
translated = simple_translate(model, test_sentence, eng_tokenizer, per_tokenizer)

print(fa(f"🔤 جمله تست: '{test_sentence}'"))
print(fa(f"🌐 ترجمه مدل: '{translated}'"))
print(fa(f"🔤 ترجمه واقعی: 'من عاشق یادگیری ماشین هستم'"))

# =============================================================================
# 9. نمایش نتایج و نمودارها
# =============================================================================
print("\n" + "="*60)
print(fa("9. نمایش نتایج و نمودارها"))
print("="*60)

# نمودار دقت و loss
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

# =============================================================================
# 10. تست مدل روی جملات جدید
# =============================================================================
print("\n" + "="*60)
print(fa("10. تست مدل روی جملات جدید"))
print("="*60)

test_sentences = [
    "Attention is important",
    "Neural networks learn",
    "AI is the future"
]

print(fa("🧪 تست مدل روی جملات جدید:"))
for sentence in test_sentences:
    translated = simple_translate(model, sentence, eng_tokenizer, per_tokenizer)
    print(fa(f"   EN: {sentence}"))
    print(fa(f"   FA: {translated}"))
    print(fa(f"   {'-'*40}"))

print(fa("\n🎉 پروژه ترجمه ماشینی با توجه کامل شد!"))