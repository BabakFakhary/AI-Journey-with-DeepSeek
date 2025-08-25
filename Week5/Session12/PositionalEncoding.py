#                                                                             به نام خدا
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
# مدل‌های ترنسفورمر به زبان پایتون ارائه می‌دهد
# اطلاعات موقعیت کلمات در جمله را به مدل می‌دهد
# ترتیب کلمات: بدون positional encoding، مدل نمی‌تواند تفاوت بین "گربه سگ را می‌زند" و "سگ گربه را می‌زند" را درک کند
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

# 2. تولید داده‌های تست
def generate_sample_data(seq_length=50, d_model=64, batch_size=16):
    # ایجاد embedding های تصادفی
    embeddings = torch.randn(batch_size, seq_length, d_model)
    return embeddings

# 3. ایجاد مدل و تست
d_model = 64      # تعداد ابعاد هر بردار
seq_length = 20   # طول جمله
batch_size = 8    # تعداد جملات در هر دسته

# ایجاد positional encoding
pos_encoder = PositionalEncoding(d_model)

# تولید داده نمونه
sample_embeddings = generate_sample_data(seq_length, d_model, batch_size)

# اعمال positional encoding
encoded_embeddings = pos_encoder(sample_embeddings)

print(fa("شکل embeddings ورودی:"), sample_embeddings.shape)
print(fa("شکل embeddings خروجی:"), encoded_embeddings.shape)

# 4. تجسم positional encoding
# نمایش گرافیکی اینکه چگونه هر موقعیت الگوی منحصر به فردی دارد
def visualize_positional_encoding(d_model=64, max_len=100):
    pe = torch.zeros(max_len, d_model)
    
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                       (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    plt.figure(figsize=(15, 5))
    
    # نمودار heatmap
    plt.subplot(1, 2, 1)
    plt.imshow(pe.numpy().T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Positional Encoding Heatmap')
    plt.xlabel(fa('موقعیت'))
    plt.ylabel(fa('بعد'))
    
    # نمودار برای چند بعد خاص
    plt.subplot(1, 2, 2)
    for i in range(0, 8, 2):
        plt.plot(pe[:, i].numpy(), label=fa(f'بعد {i}'))
    plt.title(fa('مقادیر Positional Encoding برای ابعاد مختلف'))
    plt.xlabel(fa('موقعیت'))
    plt.ylabel(fa('مقدار'))
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return pe


# تجسم
pe_matrix = visualize_positional_encoding()

# 5. تست تفاوت موقعیت‌ها
def test_position_differences():
    # مقایسه دو موقعیت مختلف
    pos1, pos2 = 5, 15
    
    plt.figure(figsize=(12, 4))
    
    # مقایسه مستقیم
    plt.subplot(1, 2, 1)
    plt.plot(pe_matrix[pos1], 'b-', label=fa(f'موقعیت {pos1}'))
    plt.plot(pe_matrix[pos2], 'r-', label=fa(f'موقعیت {pos2}'))
    plt.title(fa('مقایسه موقعیت‌های مختلف'))
    plt.xlabel(fa('بعد'))
    plt.ylabel(fa('مقدار'))
    plt.legend()
    
    # تفاوت بین موقعیت‌ها
    plt.subplot(1, 2, 2)
    difference = pe_matrix[pos1] - pe_matrix[pos2]
    plt.plot(difference, 'g-')
    plt.title(fa('تفاوت بین موقعیت‌ها'))
    plt.xlabel(fa('بعد'))
    plt.ylabel(fa('تفاوت'))
    
    plt.tight_layout()
    plt.show()

test_position_differences()

# 6. تست در یک مدل ساده
# مدل ترنسفورمر کامل
  # جریان داده:
    # کلمات → بردار
    # اضافه کردن موقعیت
    # پردازش توسط ترنسفورمر
    # طبقه‌بندی نهایی
class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(SimpleTransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # لایه Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        
        self.classifier = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # 1. تبدیل کلمات به بردار
        x = self.embedding(x)
        
        # 2. اضافه کردن اطلاعات موقعیت
        x = self.pos_encoder(x)
        
        # 3. پردازش توسط ترنسفورمر
        x = self.transformer(x)
        
        # 4. طبقه‌بندی نهایی
        x = self.classifier(x.mean(dim=1))
        return x

# 7. تست با داده واقعی
def test_with_real_data():
    # داده نمونه (شبیه‌سازی شده)
    vocab_size = 1000 # اندازه دایره واژگان
    seq_length = 20   # طول جمله
    
    # ایجاد مدل
      # یکی از مهم‌ترین پارامترهای معماری ترنسفورمر است.
      # مقایسه تعداد لایه‌ها
      #  تعداد لایه	|       پیچیدگی |	قدرت یادگیری |	زمان آموزش	| حافظه مورد نیاز  
      #           یک |              کم |         محدود |         سریع |      کم
      #        دو |           متوسط |         متوسط |        متوسط |      متوسط
      #        چهار الی شش |           زیاد |            قوی |          کند |       زیاد
      #     بیشتر از 12 |   بسیار زیاد	 |     بسیار قوی |    بسیار کند |   بسیار زیاد


    model = SimpleTransformerModel(vocab_size, d_model=64, nhead=4, num_layers=2)
    
    # داده ورودی (شبیه‌سازی شده)
    input_data = torch.randint(0, vocab_size, (8, seq_length))  # 8 نمونه، طول 20
    
    # پیش‌بینی
    output = model(input_data)
    print(fa("شکل خروجی مدل:"), output.shape)
    
    return model, output

model, output = test_with_real_data()

# 8. بررسی اهمیت Positional Encoding
def demonstrate_importance():
    # دو جمله با کلمات یکسان اما ترتیب مختلف
    sentence1 = torch.tensor([[1, 2, 3, 4]])  # "گربه سگ پرید میوه"
    sentence2 = torch.tensor([[4, 3, 2, 1]])  # "میوه پرید سگ گربه"
    
    # بدون positional encoding
    embedding = nn.Embedding(10, 64)
    emb1 = embedding(sentence1)
    emb2 = embedding(sentence2)
    
    # با positional encoding
    pos_emb1 = pos_encoder(emb1)
    pos_emb2 = pos_encoder(emb2)
    
    # محاسبه شباهت
    similarity_no_pos = torch.cosine_similarity(emb1.flatten(), emb2.flatten(), dim=0)
    similarity_with_pos = torch.cosine_similarity(pos_emb1.flatten(), pos_emb2.flatten(), dim=0)
    
    # استفاده از f-string برای حل مشکل
    print(fa(f"شباهت بدون Positional Encoding: {similarity_no_pos.item():.4f}"))
    print(fa(f"شباهت با Positional Encoding: {similarity_with_pos.item():.4f}"))
    
    # تفاوت باید قابل توجه باشد!
    difference = abs(similarity_no_pos - similarity_with_pos)
    print(fa(f"تفاوت: {difference.item():.4f}"))
    
    if difference > 0.1:
        print(fa("✅ Positional Encoding به درستی کار می‌کند!"))
    else:
        print(fa("⚠️  Positional Encoding تاثیر کمی دارد"))

demonstrate_importance()

# 9. کاربردهای عملی
print(fa("\n🎯 کاربردهای Positional Encoding:"))
applications = [
    "مدل‌های ترجمه ماشینی",
    "چت‌بات‌های هوشمند",
    "پردازش زبان طبیعی",
    "تولید متن",
    "خلاصه‌سازی خودکار",
    "پاسخ به سوالات"
]

for i, app in enumerate(applications, 1):
    print(f"{i}. {fa(app)}")

# 10. ذخیره و بارگذاری
def save_and_load_example():
    # ذخیره مدل
    torch.save({
        'model_state_dict': model.state_dict(),
        'pos_encoder_state_dict': pos_encoder.state_dict()
    }, 'positional_encoding_model.pth')
    
    print(fa("مدل ذخیره شد!"))
    
    # بارگذاری مدل
    checkpoint = torch.load('positional_encoding_model.pth')
    
    # ایجاد مدل جدید
    new_model = SimpleTransformerModel(1000, 64, 4, 2)
    new_model.load_state_dict(checkpoint['model_state_dict'])
    
    new_pos_encoder = PositionalEncoding(64)
    new_pos_encoder.load_state_dict(checkpoint['pos_encoder_state_dict'])
    
    print(fa("مدل بارگذاری شد!"))

save_and_load_example()
