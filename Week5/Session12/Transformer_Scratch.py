#                                                                       به نام خدا
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------
# این کد یک لایه  (خودتوجهی) را از صفر پیاده‌سازی می‌کند 
# هدف کد: درک اینکه مدل‌های زبانی چگونه به کلمات "توجه" می‌کنند
#----------------------------------------------------------------------------------

# --------------------------------------
# تعریف کلاس SelfAttention (خودتوجهی)
# --------------------------------------
class SelfAttention(tf.keras.layers.Layer):
    # سازنده (init)
    def __init__(self, d_model):        
        super(SelfAttention, self).__init__()
        # d_model: بعد بردارهای داخلی (مثلاً ۶۴ بعد)
        self.d_model = d_model
        # سه لایه Dense برای تولید:
        # Query (wq): چیزی که می‌خواهم بدانم
        self.wq = tf.keras.layers.Dense(d_model)
        # Key (wk): چیزی که دارم
        self.wk = tf.keras.layers.Dense(d_model)
        # Value (wv): اطلاعات واقعی
        self.wv = tf.keras.layers.Dense(d_model)
    
    # --------------------------------------
    # عملیات اصلی (call)
    # --------------------------------------
    # ورودی: یک جمله به صورت بردار (مثلاً ۵ کلمه، هر کلمه ۶۴ بعد)
    def call(self, inputs):

        # مثال در دنیای واقعی
          # Query : من چه چیزی نیاز دارم بدانم؟" (مثلاً: ضمیر "او" به چه اشاره دارد؟)
          # Key: "چه اطلاعاتی دارم؟" (همه کلمات جمله)
          # Value: "اطلاعات واقعی" (معنی هر کلمه)

        Q = self.wq(inputs) # تولید Query
        K = self.wk(inputs) # تولید Key 
        V = self.wv(inputs) # تولید Value      

        # محاسبه توجه
          # مرحله ۱: محاسبه شباهت بین هر جفت کلمه (ضرب Q و K)
          # مرحله ۲: نرمال‌سازی برای پایداری عددی
          # مرحله ۳: تبدیل به احتمال با softmax
          # مرحله ۴: ترکیب weighted با مقادیر واقعی (V)
        scores = tf.matmul(Q, K, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # خروجی: همان جمله اما با "توجه" به روابط بین کلمات
        return tf.matmul(attention_weights, V)

# --------------------------------------
# تست اصلی با ابعاد صحیح
# --------------------------------------
# ساخت یک ورودی نمونه (Dummy Input)
  # tf.random.normal: تولید اعداد تصادفی از توزیع نرمال  
  # مثال واقعی: مثل اینه که یک جمله ۵ کلمه‌ای داریم و هر کلمه با ۶۴ عدد نمایش داده شد
  # (1, 5, 64): شکل (shape) ورودی → ۱ جمله، ۵ کلمه، هر کلمه ۶۴ ویژگی 
sample_input = tf.random.normal((1, 5, 64))  

# ایجاد لایه Self-Attention
  # 64: پارامتر d_model → باید با بعد آخر ورودی یکسان باشد (۶۴) 
attention_layer = SelfAttention(64)

# اعمال لایه روی ورودی
  # ورودی را به لایه Self-Attention می‌دهیم
  # لایه محاسبات توجه را انجام می‌دهد
  # خروجی جدیدی تولید می‌کند
output = attention_layer(sample_input)

print("Input shape:", sample_input.shape)
print("Output shape:", output.shape)

# --------------------------------------
# تست دوم با ابعاد متناسب
# --------------------------------------
# ساخت داده تست ساده
  # می‌سازیم یک جمله مصنوعی با ۳ کلمه
  # هر کلمه یک بردار ۲ بعدی داره (مثلاً [1.0, 0.5])
  # مثل اینه: ["گربه", "خواب", "نرم"]
test_input = tf.constant([[[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]]], dtype=tf.float32)

# تکثیر برای batch
  # مدل‌های واقعی همزمان چند جمله پردازش می‌کنن (batch)
  # اینجا ۲ تا جمله یکسان می‌سازیم
test_input = tf.tile(test_input, [2, 1, 1])  # شکل: (2, 3, 2)

# محاسبه توجه
  # محاسبه می‌کنه هر کلمه چقدر به کلمات دیگه شباهت داره
attention_layer_2 = SelfAttention(d_model=2)  # d_model متناسب با ورودی
Q = attention_layer_2.wq(test_input)
K = attention_layer_2.wk(test_input)
scores = tf.matmul(Q, K, transpose_b=True)
# تبدیل به احتمال
  # اعداد رو به احتمال تبدیل می‌کنه (بین ۰ تا ۱)
  # جمع هر سطر = ۱ (مثل درصد توجه)
attention_weights = tf.nn.softmax(scores, axis=-1)

# --------------------------------------
# توضیح نمودار Heatmap:
# --------------------------------------
# تفسیر نمودار 
  # مثال ساده
  # فرض کنید جمله "من به مغازه رفتم" را داریم:
  # اگر مدل روی کلمه "رفتم" (موقعیت ۳) تمرکز کند
  # و بیشترین توجه به "من" (موقعیت ۱) داشته باشد
  # در نمودار، نقطه (۳,۱) روشن خواهد بود
# خواندن نمودار:
  # محور عمودی (Query): کلمه‌ای که سؤال داره ("چه کلمه‌ای به من توجه کنه؟")
  # محور افقی (Key): کلمه‌ای که جواب میده ("من چقدر مهمم؟")
  # رنگ‌ها: میزان توجه (زرد = توجه زیاد، بنفش = توجه کم)
# Heatmap توجه:
 # محور X: کلمات کلیدی (Key)
 # محور Y: کلمات پرسشی (Query)
 # رنگ روشن: توجه بیشتر
plt.figure(figsize=(10, 5))
plt.imshow(attention_weights[0].numpy(), cmap='viridis')
plt.colorbar()
plt.title('Attention Weights')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.show()

