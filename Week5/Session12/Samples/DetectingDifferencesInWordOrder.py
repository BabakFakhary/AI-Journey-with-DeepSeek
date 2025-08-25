#                                                                     به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import torch  #  کتابخانه اصلی یادگیری عمیق
import torch.nn as nn
import numpy as np # محاسبات عددی
import matplotlib.pyplot as plt 
import math

# --------------------------------------------------------------------------------------------------------
# تشخیص تفاوت ترتیب کلمات
# --------------------------------------------------------------------------------------------------------

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

# 1. کلاس Positional Encoding
# این الگو به مدل کمک می‌کند هم موقعیت مطلق و هم موقعیت نسبی کلمات را یاد بگیرد
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # ایجاد ماتریس positional encoding
        pe = torch.zeros(max_len, d_model)
        
        #  # محاسبه موقعیت‌ها (0 تا max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # محاسده divisor term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # اعمال سینوس و کسینوس
          # این کلاس از فرمول سینوس و کسینوس استفاده می‌کند تا به هر موقعیت در جمله یک امضای عددی منحصر به فرد بدهد.
        pe[:, 0::2] = torch.sin(position * div_term)  # برای موقعیت‌های زوج از سینوس استفاده می‌کند
        pe[:, 1::2] = torch.cos(position * div_term)  # برای موقعیت‌های فرد از کسینوس استفاده می‌کند
        
        # اضافه کردن بعد batch
        pe = pe.unsqueeze(0)
        
        # ثبت به عنوان buffer (نه parameter)
        self.register_buffer('pe', pe)
    
    # این تابع اطلاعات موقعیت را به بردارهای کلمات اضافه می‌کند
    def forward(self, x):
        # اضافه کردن positional encoding به embedding
        return x + self.pe[:, :x.size(1)]

# 2. ایجاد مدل ساده
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        return x

# 3. تست عملی
def test_real_example():
    print(fa("🎯 مثال عملی: تشخیص تفاوت ترتیب کلمات"))
    print(fa("=" * 50))
    
    # ایجاد مدل
    vocab_size = 10  # 10 کلمه مختلف
    d_model = 8     # ابعاد کوچک برای سادگی
    model = SimpleModel(vocab_size, d_model)
    
    # دو جمله با کلمات یکسان اما ترتیب مختلف
    # فرض کنید: 1=گربه, 2=سگ, 3=موش, 4=پرید
    sentence1 = torch.tensor([[1, 2, 3, 4]])  # "گربه سگ موش پرید"
    sentence2 = torch.tensor([[4, 3, 2, 1]])  # "پرید موش سگ گربه"
    
    print(fa("جمله ۱: گربه سگ موش پرید"))
    print(fa("جمله ۲: پرید موش سگ گربه"))
    print(fa("کدهای جمله ۱:"), sentence1.tolist())
    print(fa("کدهای جمله ۲:"), sentence2.tolist())
    print()
    
    # پردازش بدون Positional Encoding
     # یعنی: فقط از قسمت embedding مدل استفاده کن
    embeddings_only = model.embedding(sentence1)
    print(fa("📊 بدون Positional Encoding:"))
    print(fa("شکل embeddings:"), embeddings_only.shape)
    print(fa("مقادیر نمونه (اولین کلمه):"))
    print(embeddings_only[0, 0, :4].detach().numpy())  # فقط 4 بعد اول
    print()
    
    # پردازش با Positional Encoding
     # یعنی: از کل مدل استفاده کن (هم embedding + هم positional encoding)
    with_pos_encoding = model(sentence1)
    print(fa("📊 با Positional Encoding:"))
    print(fa("شکل خروجی:"), with_pos_encoding.shape)
    print(fa("مقادیر نمونه (اولین کلمه):"))
    print(with_pos_encoding[0, 0, :4].detach().numpy())  # فقط 4 بعد اول
    print()

# 4. محاسبه شباهت بین دو جمله
    def calculate_similarity():
        # پردازش هر دو جمله
        emb1 = model(sentence1)
        emb2 = model(sentence2)
        
        # محاسبه شباهت کسینوسی
          # ک تابع بسیار مهم و کاربردی است که میزان شباهت بین دو بردار را محاسبه می‌کند
          # این تابع کسینوس زاویه بین دو بردار را محاسبه می‌کند. هر چه این مقدار به ۱ نزدیک‌تر باشد، دو بردار شبیه‌تر هستند
        similarity = torch.cosine_similarity(
            emb1.flatten(), 
            emb2.flatten(), 
            dim=0
        )
        
        return similarity.item()
    
# 5. تست چندباره برای درک بهتر
    print(fa("🔍 مقایسه دو جمله:"))
    for i in range(3):
        similarity = calculate_similarity()
        print(fa(f"شباهت بین دو جمله: {similarity:.4f}"))
    
    print()
    print(fa("✅ نتیجه: هر چه شباهت کمتر باشد، مدل بهتر تفاوت ترتیب را تشخیص می‌دهد!"))

# 6. تجسم Positional Encoding برای درک بهتر
# این تابع می‌خواهد نشان دهد که چگونه هر موقعیت در جمله یک امضای عددی منحصر به فرد دارد
def visualize_simple_pe():
    print(fa("\n👀 تجسم Positional Encoding برای 5 موقعیت و 4 بعد:"))
    
    # ایجاد PE برای ابعاد کوچک
     # ۵ موقعیت: یعنی ۵ کلمه در جمله
     # ۴ بعد: یعنی هر کلمه با ۴ عدد نمایش داده می‌شود (ساده شده)
    pe = torch.zeros(5, 4)  # 5 موقعیت، 4 بعد
    position = torch.arange(0, 5, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, 4, 2).float() * (-math.log(10000.0) / 4))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # خروجی ماتریس
    
      # سطرها: موقعیت‌های کلمات (0 تا 4)
      # ستون‌ها: ابعاد مختلف (0 تا 3)
      # مقادیر: بین -1 تا 1
  
      # برای موقعیت ۰ (اولین کلمه):
      #[ 0.00,  1.00,  0.00,  1.00]  # بسیار ساده
  
      # برای موقعیت ۲ (سومین کلمه):  
      # [ 0.91, -0.42,  0.0002, 0.999998]  # پیچیده‌تر
      
      # برای موقعیت ۴ (پنجمین کلمه):
      # [-0.76, -0.65,  0.0004, 0.999992]  # کاملاً متفاوت

    print(fa("ماتریس Positional Encoding (5 موقعیت × 4 بعد):"))
    print(pe.numpy())
    print()
    
    # رسم ساده
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(pe.numpy(), cmap='viridis', aspect='auto')
    plt.title(fa('Heatmap Positional Encoding'))
    plt.xlabel(fa('بعد'))
    plt.ylabel(fa('موقعیت'))
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    for i in range(4):
        plt.plot(pe[:, i].numpy(), label=fa(f'بعد {i}'))
    plt.title(fa('مقادیر برای هر بعد'))
    plt.xlabel(fa('موقعیت'))
    plt.ylabel(fa('مقدار'))
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 7. اجرای مثال
if __name__ == "__main__":
    test_real_example()
    visualize_simple_pe()
    
    print(fa("\n" + "=" * 50))
    print(fa("🎓 خلاصه مفهوم:"))
    print(fa("• بدون Positional Encoding: همه جملات شبیه هم دیده می‌شوند"))
    print(fa("• با Positional Encoding: ترتیب کلمات تشخیص داده می‌شود"))
    print(fa("• هر موقعیت امضای عددی منحصر به فرد دارد"))
    print(fa("• مدل می‌فهمد 'گربه سگ را می‌زند' ≠ 'سگ گربه را می‌زند'"))