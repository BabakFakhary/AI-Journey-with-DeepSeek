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
#  حداقل 1000-10000 جمله نیاز داریم
# -------------------------------------------------------------------------------------------------

# =============================================================================
# 1. ایجاد دیتاست نمونه (انگلیسی به فارسی)
# =============================================================================
print("\n" + "="*60)
print(fa("1. ایجاد دیتاست نمونه (انگلیسی به فارسی)"))
print("="*60)

# دیتاست گسترده‌تر با 50 جمله
english_sentences = [
    # جملات پایه
    "I love machine learning",
    "Attention is important",
    "Transformers are powerful",
    "Neural networks learn",
    "Deep learning is amazing",
    "AI is the future",
    "Natural language processing",
    "Computer vision applications",
    "Hello world program",
    "GPT models are large",
    "Data science is interesting",
    "Python is popular",
    "TensorFlow is a framework",
    "PyTorch is also good",
    "Mathematics is fundamental",
    "Algorithms are essential",
    "Programming requires practice",
    "Open source is great",
    "Cloud computing is scalable",
    "Big data analytics",
    
    # جملات مربوط به یادگیری ماشین
    "Training a model takes time",
    "Validation improves accuracy",
    "Testing is crucial",
    "Overfitting is a problem",
    "Underfitting also bad",
    "Gradient descent optimization",
    "Backpropagation algorithm",
    "Convolutional neural networks",
    "Recurrent neural networks",
    "Supervised learning examples",
    
    # جملات کاربردی
    "The model predicts results",
    "Accuracy measures performance",
    "Precision and recall metrics",
    "F1 score balance",
    "Confusion matrix visualization",
    "Feature engineering important",
    "Data preprocessing necessary",
    "Normalization improves training",
    "Regularization prevents overfitting",
    "Hyperparameter tuning optimization",
    
    # جملات پیشرفته‌تر
    "Transfer learning useful",
    "Fine tuning models",
    "Embedding layers represent words",
    "Sequence to sequence models",
    "Attention mechanism powerful",
    "BERT transformer architecture",
    "GPT generative model",
    "Reinforcement learning different",
    "Unsupervised learning clustering",
    "Semi supervised learning"
]

persian_sentences = [
    # جملات پایه
    "من عاشق یادگیری ماشین هستم",
    "توجه مهم است", 
    "ترانسفورمرها قدرتمند هستند",
    "شبکه های عصبی یاد می گیرند",
    "یادگیری عمیق شگفت انگیز است",
    "هوش مصنوعی آینده است",
    "پردازش زبان طبیعی",
    "کاربردهای بینایی کامپیوتر",
    "برنامه سلام دنیا",
    "مدل های جی پی تی بزرگ هستند",
    "علم داده جذاب است",
    "پایتون محبوب است",
    "تنسورفلو یک فریمورک است",
    "پای تورچ نیز خوب است",
    "ریاضیات پایه است",
    "الگوریتم ها ضروری هستند",
    "برنامه نویسی نیاز به تمرین دارد",
    "متن باز عالی است",
    "رایانش ابری مقیاس پذیر است",
    "تحلیل داده های بزرگ",
    
    # جملات مربوط به یادگیری ماشین
    "آموزش یک مدل زمان می برد",
    "اعتبارسنجی دقت را بهبود می دهد",
    "تست بسیار مهم است",
    "اورفیتینگ یک مشکل است",
    "آندر فیتینگ نیز بد است",
    "بهینه سازی نزول گرادیان",
    "الگوریتم پس انتشار",
    "شبکه های عصبی کانولوشنی",
    "شبکه های عصبی بازگشتی",
    "مثال های یادگیری نظارت شده",
    
    # جملات کاربردی
    "مدل نتایج را پیش بینی می کند",
    "دقت عملکرد را اندازه می گیرد",
    "معیارهای دقت و فراخوانی",
    "تعادل امتیاز اف یک",
    "نمایش ماتریس درهمی",
    "مهندسی ویژگی مهم است",
    "پیش پردازش داده ضروری است",
    "نرمال سازی آموزش را بهبود می دهد",
    "منظم سازی از اورفیتینگ جلوگیری می کند",
    "بهینه سازی تنظیم هایپرپارامتر",
    
    # جملات پیشرفته‌تر
    "یادگیری انتقالی مفید است",
    "تنظیم دقیق مدل ها",
    "لایه های تعبیه کلمات را نشان می دهند",
    "مدل های دنباله به دنباله",
    "مکانیزم توجه قدرتمند است",
    "معماری ترانسفورمر برت",
    "مدل مولد جی پی تی",
    "یادگیری تقویتی متفاوت است",
    "خوشه بندی یادگیری بدون نظارت",
    "یادگیری نیمه نظارت شده"
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
# 5. ساخت مدل ترجمه sequence-to-sequence با مکانیزم توجه
# =============================================================================
print("\n" + "="*60)
print(fa("5. ساخت مدل ترجمه sequence-to-sequence با مکانیزم توجه"))
print("="*60)

class Seq2SeqTranslationModel(tf.keras.Model):
    def __init__(self, vocab_size_eng, vocab_size_per, d_model, num_heads):
        super(Seq2SeqTranslationModel, self).__init__()
        
        self.eng_embedding = tf.keras.layers.Embedding(vocab_size_eng, d_model)
        self.per_embedding = tf.keras.layers.Embedding(vocab_size_per, d_model)
        
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.decoder_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, num_heads)
        
        self.final_dense = tf.keras.layers.Dense(vocab_size_per, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(0.1)
        
    def call(self, inputs, training=False):
        eng_input, per_input = inputs
        
        # رمزگذار (Encoder): ورودی را پردازش می‌کند
          # Encoder مثل یک مترجم است که
            # 🔍 متن خارجی را می‌خواند
            # 🧠 مفهوم آن را می‌فهمد
            # 📝 یادداشت برمی‌دارد
          #  ورودی: "The cat sat on the mat"
            # دریافت جمله انگلیسی
            # تبدیل به توکن: ["The", "cat", "sat", "on", "the", "mat"]
            # محاسبه Self-Attention: "cat" به "sat" مرتبط است
            # ایجاد representation کلی
        eng_embedded = self.eng_embedding(eng_input)
        enc_output, _ = self.encoder_attention(eng_embedded, eng_embedded, eng_embedded)
        enc_output = self.dropout(enc_output, training=training)
        
        # رمزگشا (Decoder): خروجی را تولید می‌کند
          # مثل یک نویسنده است
           # 📖 یادداشت‌های مترجم را می‌خواند
           # 🎨 داستان را به زبان جدید می‌نویسد
           # ✍️ کلمه به کلمه پیش می‌رود
          # مراحل Decoder
            #  شروع با توکن شروع: "<start>"
            # نگاه به Encoder: "چه کلمه‌ای اول بیاید؟"
            # تولید: "گربه"
            # تکرار: "<start> گربه" → به Encoder نگاه کن → "نشست"
            # 🏁 ادامه تا تولید جمله کامل: "گربه روی فرش نشست"
        per_embedded = self.per_embedding(per_input)
        dec_output, _ = self.decoder_attention(per_embedded, per_embedded, per_embedded)
        dec_output = self.dropout(dec_output, training=training)
        
        # توجه بین رمزگذار و رمزگشا
         # پل ارتباطی بین این دو است
        context, attention_weights = self.encoder_decoder_attention(
            dec_output, enc_output, enc_output)
        context = self.dropout(context, training=training)
        
        # پیش‌بینی نهایی برای هر موقعیت زمانی
        output = self.final_dense(context)
        
        return output

# پارامترهای مدل
vocab_size_eng = eng_vocab_size
vocab_size_per = per_vocab_size
d_model = 128  # افزایش بعد embedding
num_heads = 4

print(fa(f"🔧 پارامترهای مدل:"))
print(fa(f"   - vocab_size_eng: {vocab_size_eng}"))
print(fa(f"   - vocab_size_per: {vocab_size_per}"))
print(fa(f"   - d_model: {d_model}"))
print(fa(f"   - num_heads: {num_heads}"))

# ایجاد مدل
model = Seq2SeqTranslationModel(vocab_size_eng, vocab_size_per, d_model, num_heads)
print(fa("✅ مدل ترجمه sequence-to-sequence ایجاد شد"))

# =============================================================================
# 6. تست مدل با داده‌های نمونه
# =============================================================================
print("\n" + "="*60)
print(fa("6. تست مدل با داده‌های نمونه"))
print("="*60)

sample_idx = 0
sample_eng = tf.constant([eng_padded[sample_idx]])
sample_per = tf.constant([per_padded[sample_idx]])

print(fa(f"🔍 تست مدل با نمونه {sample_idx + 1}:"))
print(fa(f"   EN: {english_sentences[sample_idx]}"))
print(fa(f"   FA: {persian_sentences[sample_idx]}"))

# اجرای مدل
output = model([sample_eng, sample_per])

print(fa(f"\n📊 خروجی مدل:"))
print(fa(f"   Output shape: {output.shape}"))  # باید (1, max_len, vocab_size_per) باشد

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

# آماده‌سازی داده‌های آموزشی برای sequence-to-sequence
# برای آموزش، از جمله فارسی با یک موقعیت تأخیر استفاده می‌کنیم
decoder_input_data = per_padded[:, :-1]  # همه به جز آخرین توکن
decoder_target_data = per_padded[:, 1:]   # همه به جز اولین توکن

# تقسیم داده
(X_train, X_val, 
 decoder_input_train, decoder_input_val,
 decoder_target_train, decoder_target_val) = train_test_split(
    eng_padded, decoder_input_data, decoder_target_data, 
    test_size=0.2, random_state=42
)

print(fa(f"📊 داده‌های آموزش: {X_train.shape[0]} نمونه"))
print(fa(f"📊 داده‌های validation: {X_val.shape[0]} نمونه"))

# آموزش مدل
print(fa("\n🎯 شروع آموزش مدل..."))
history = model.fit(
    [X_train, decoder_input_train],  # ورودی encoder و decoder
    decoder_target_train,            # هدف‌ها (یک موقعیت بعدی)
    epochs=300,
    batch_size=4,
    validation_data=(
        [X_val, decoder_input_val],
        decoder_target_val
    ),
    verbose=1
)

print(fa("✅ آموزش مدل کامل شد"))

# =============================================================================
# 8. توابع کمکی برای ترجمه کامل جمله
# =============================================================================
print("\n" + "="*60)
print(fa("8. توابع کمکی برای ترجمه کامل جمله"))
print("="*60)

def translate_sentence(model, sentence, eng_tokenizer, per_tokenizer, max_len=10):
    """ترجمه کامل یک جمله انگلیسی به فارسی"""
    try:
        # پیش‌پردازش جمله ورودی
        sequence = eng_tokenizer.texts_to_sequences([sentence])
        encoder_input = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=max_len, padding='post')
        
        # شروع با توکن شروع (استفاده از اولین توکن فارسی)
        start_token = list(per_tokenizer.word_index.values())[0] if per_tokenizer.word_index else 1
        decoder_input = np.array([[start_token] + [0]*(max_len-1)])
        
        # تولید ترجمه به صورت حریصانه
        translated_tokens = []
        
        for i in range(max_len):
            # پیش‌بینی توکن بعدی
            predictions = model([tf.constant(encoder_input), tf.constant(decoder_input)])
            predicted_id = tf.argmax(predictions[0, i, :]).numpy()
            
            if predicted_id == 0:  # توکن padding
                break
                
            translated_tokens.append(predicted_id)
            
            # به روزرسانی ورودی دیکدر برای مرحله بعد
            if i < max_len - 1:
                decoder_input[0, i + 1] = predicted_id
        
        # تبدیل توکن‌ها به متن
        translated_words = []
        for token_id in translated_tokens:
            word = per_tokenizer.index_word.get(token_id, '')
            if word and word != '<OOV>':
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    except Exception as e:
        return f"خطا در ترجمه: {e}"

# تست تابع ترجمه
test_sentence = "I love machine learning"
translated = translate_sentence(model, test_sentence, eng_tokenizer, per_tokenizer)

print(fa(f"🔤 جمله تست: '{test_sentence}'"))
print(fa(f"🌐 ترجمه مدل: '{translated}'"))
print(fa(f"🔤 ترجمه واقعی: 'من عاشق یادگیری ماشین هستم'"))

# =============================================================================
# 9. نمایش نتایج و نمودارها
# =============================================================================
print("\n" + "="*60)
print(fa("9. نمایش نتایج و نمودارها"))
print("="*60)

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
    "AI is the future",
    "Deep learning is amazing"
]

print(fa("🧪 تست مدل روی جملات جدید:"))
for sentence in test_sentences:
    translated = translate_sentence(model, sentence, eng_tokenizer, per_tokenizer)
    print(fa(f"   EN: {sentence}"))
    print(fa(f"   FA: {translated}"))
    print(fa(f"   {'-'*40}"))

print(fa("\n🎉 پروژه ترجمه ماشینی با توجه کامل شد!"))