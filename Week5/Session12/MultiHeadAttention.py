#                                                                     به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install numpy scikit-learn matplotlib  torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------------------------------
# این مکانیزمی است که به مدل اجازه می‌دهد همزمان به جنبه‌های مختلف اطلاعات توجه کند.
# ---------------------------------------------------------------------------------------------------

# -------------------------------------------
#  تعریف پارامترها
# -------------------------------------------

# d_model: بعد بردارهای embedding (64 بعد)
d_model = 64  # بعد embedding

# num_heads: تعداد headهای attention (8 head)
num_heads = 8  # تعداد headهای attention

# seq_length: طول دنباله ورودی (10 کلمه/توکن)
seq_length = 10  # طول sequence

# batch_size: تعداد نمونه‌ها در هر batch (32 نمونه)
batch_size = 32  # اندازه batch


# ----------------------------------------------
# MultiHeadAttention
# ----------------------------------------------

# توضیح: تعریف کلاس اصلی که از nn.Module ارث‌بری می‌کند
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads        
        self.d_k = d_model // num_heads # d_k: بعد هر head (64/8 = 8)
        
        # لایه‌های خطی برای Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    # بخش چهارم: تابع forward (توضیح: تابعی که محاسبات را انجام می‌دهد)
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # تبدیل به Q, K, V
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # محاسبه attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # اعمال mask (اگر وجود دارد)
        # توضیح: اگر mask وجود دارد، مقادیر masked را با عدد بسیار کوچک جایگزین می‌کند
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # محاسبه attention weights
        #توضیح: اعمال تابع softmax برای تبدیل scores به وزن‌های attention (جمع هر سطر=1)
        attention_weights = F.softmax(scores, dim=-1)

        # ضرب وزن‌ها در values
        output = torch.matmul(attention_weights, V)

        # الحاق headها
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # لایه خروجی
        output = self.W_o(output)
        
        return output, attention_weights


# تولید داده تست
def generate_test_data(batch_size, seq_length, d_model):
    # داده‌های تصادفی
    X = torch.randn(batch_size, seq_length, d_model)
    
    # برچسب‌های ساده (جمع عناصر در بعد embedding)
    y = (torch.sum(X, dim=[1, 2]) > 0).long()
    
    return X, y

# ایجاد مدل
model = MultiHeadAttention(d_model, num_heads)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# آموزش مدل
losses = []
accuracies = []
attention_weights_history = []

# حلقه آموزش
for epoch in range(100):
    # تولید داده
    X, y = generate_test_data(batch_size, seq_length, d_model)
    
    # پیش‌بینی مدل
    # چرا از یک ماتریس سه بار استفاده می‌کنیم
      # Query: چیزی که می‌پرسیم ("گربه به چه چیزی مربوط است؟")
      # Key: چیزی که با آن مقایسه می‌کنیم ("آیا 'خوابید' به 'گربه' مربوط است؟")
      # Value: اطلاعاتی که می‌خواهیم استخراج کنیم ("اطلاعات مربوط به 'خوابید'")
    output, attention_weights = model(X, X, X)
    
    # محاسبه loss (تبدیل خروجی به شکل مناسب)
    logits = output.mean(dim=1)  # میانگین گیری روی sequence
    loss = criterion(logits, y)
    
    # محاسبه accuracy
    predictions = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(y.cpu().numpy(), predictions.cpu().detach().numpy())
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # ذخیره مقادیر
    losses.append(loss.item())
    accuracies.append(accuracy)
    attention_weights_history.append(attention_weights.detach())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# رسم نمودارها
plt.figure(figsize=(15, 5))

# نمودار loss
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# نمودار accuracy
plt.subplot(1, 3, 2)
plt.plot(accuracies)
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# نمودار attention weights برای آخرین batch
plt.subplot(1, 3, 3)
# میانگین وزن‌های attention برای head اول - بدون گرفتن mean
avg_attention = attention_weights_history[-1][0, 0].cpu().numpy()
plt.imshow(avg_attention, cmap='hot', interpolation='nearest')
plt.title('Attention Weights (Head 1)')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.colorbar()

plt.tight_layout()
plt.show()

# نمایش جزئیات وزن‌های attention برای یک نمونه
print(get_display(arabic_reshaper.reshape("\nنمونه‌ای از وزن‌های attention (برای head اول):")))
print(attention_weights_history[-1][0, 0, :, :])  # [batch, head, query, key]


