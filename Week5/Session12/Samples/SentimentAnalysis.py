#                                                                              به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from collections import Counter

# ---------------------------------------------------------------------------------
# پروژه کوچک: تحلیل احساسات (Sentiment Analysis)
# اگر هدف شما یادگیری مفاهیم پایه Attention و داشتن کنترل کامل است، فایل اول (PyTorch) گزینه بسیار خوبی است.
# نوع مدل : یک لایه Multi-Head Attention ساده
# Embedding -> Attention -> Classifier
# ---------------------------------------------------------------------------------

# ----------------------------
# ۱. آماده‌سازی داده‌های واقعی
# ----------------------------
# داده‌های نمونه از نظرات کاربران
reviews = [
    "این محصول عالی است و کیفیت فوق العاده‌ای دارد",
    "خیلی بد بود، پولم رو هدر دادم",
    "متوسط بود، نه خوب نه بد",
    "عالی عالی عالی، واقعا راضیم",
    "اصلا خوب نبود، توصیه نمی‌کنم",
    "خیلی خوبه، ارزش خرید داره",
    "بدترین خرید عمرم بود",
    "قیمت مناسبی داره و کیفیت خوبیه",
    "نمی‌ارزه به این قیمت",
    "عالیه، حتما again میخرم",
    "خیلی ضعیف، زود خراب شد",
    "راضی هستم، خوب کار می‌کنه",
    "بدترین محصول ممکن",
    "کیفیت عالی، بسته‌بندی زیبا",
    "مناسب نیست، بهتره نخرید",
    "خیلی سبک و با کیفیت",
    "گرون نیست به این خوبی",
    "افتضاح بود، پشیمونم",
    "عالی بود، ممنون از فروشگاه",
    "نه خوبه نه بد، متوسط"
]

# برچسب‌های احساسات (0: منفی, 1: مثبت, 2: خنثی)
labels = [1, 0, 2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 2]

# ----------------------------
# ۲. پیش‌پردازش متن
# ----------------------------

def preprocess_text(text):
    # حذف علائم نگارشی و اعداد
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # تبدیل به حروف کوچک
    text = text.lower()
    # حذف فاصله‌های اضافی
    text = ' '.join(text.split())
    return text

# پیش‌پردازش همه نظرات
processed_reviews = [preprocess_text(review) for review in reviews]

# ایجاد دیکشنری لغات
  # Counter(): یک شیء می‌سازد که کلمات را می‌شمارد
  # خروجی : Counter({'عالی': 3, 'بد': 2, 'خوب': 2, 'کیفیت': 2, ...})
word_counts = Counter()
for review in processed_reviews:
    words = review.split()    # review.split(): هر نظر را به کلمات جداگانه تقسیم می‌کند
    word_counts.update(words) # تعداد هر کلمه را افزایش می‌دهد

# ایجاد vocabulary
  # enumerate(word_counts.items()): به هر کلمه یک index می‌دهد
  # شماره  کلمه از ۲ شروع می‌شود 
  # {word: idx+2}: یک دیکشنری می‌سازد که هر کلمه را به یک نسبت می‌کند
  # { 'عالی': 2,'بد': 3, 'خوب': 4,'کیفیت': 5,...}
  # چرا از ۲ شروع می‌کنیم : چون می‌خواهیم ۰ و ۱ را برای توکن‌های خاص نگه داریم
  # فرض کنید این جمله جدید داریم: "این محصول عالی است 
    # کلمات: ["این", "محصول", "عالی", "است"]
    #  اگر "این" و "است" در vocabulary نباشند:
    # sequence = [vocab.get("این", vocab['<UNK>']),    # -> 1 (ناشناخته)   
    #             vocab.get("محصول", vocab['<UNK>']),  # -> 1 (ناشناخته)   
    #             vocab.get("عالی", vocab['<UNK>']),   # -> 2 (مشخص است)  
    #              vocab.get("است", vocab['<UNK>'])     # -> 1 (ناشناخته)]
vocab = {word: idx+2 for idx, (word, count) in enumerate(word_counts.items())}
vocab['<PAD>'] = 0  # برای padding
vocab['<UNK>'] = 1  # برای کلمات ناشناخته

vocab_size = len(vocab)
print(get_display(arabic_reshaper.reshape(f"اندازه دیکشنری: {vocab_size}")))
print(get_display(arabic_reshaper.reshape("نمونه‌ای از لغات:")))
print(' '.join([get_display(arabic_reshaper.reshape(word)) for word in list(vocab.keys())[:10]]))

# ----------------------------
# ۳. تبدیل متن به اعداد
# ----------------------------
# این تابع لغات رو به عدد تبدیل می کند و در ادامه در صورتی که طول کمتر از ماکس لنث باشد کاراکتر جای خالی وگرنه خود متن تا 10 کلمه رو برمی گرداند
def text_to_sequence(text, vocab, max_length=10):
    words = text.split()
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
    # padding یا truncate
    if len(sequence) < max_length:
        sequence = sequence + [vocab['<PAD>']] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

# تبدیل همه نظرات به sequences
max_length = 8
sequences = [text_to_sequence(review, vocab, max_length) for review in processed_reviews]

# ----------------------------
# ۴. ایجاد مدل تحلیل احساسات
# ----------------------------
class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_classes):
        super(SentimentAnalysisModel, self).__init__()
        
        # لایه embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # لایه attention
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        
        # لایه‌های طبقه‌بندی
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # تبدیل کلمات به بردار
        embeddings = self.embedding(x)
        
        # اعمال attention
        output, attn_weights = self.attention(embeddings, embeddings, embeddings)
        
        # میانگین گیری از خروجی sequence
        pooled_output = output.mean(dim=1)
        
        # طبقه‌بندی
        logits = self.classifier(pooled_output)
        return logits, attn_weights
    
# ----------------------------
# ۵. MultiHeadAttention (همان کد قبلی)
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output, attention_weights
    
# ----------------------------
# ۶. آماده‌سازی داده برای آموزش
# ----------------------------
# تانسور یک ساختار داده‌ای است 
  # مانند لیست یا آرایه عمل می‌کند
  # می‌تواند ابعاد مختلف داشته باشد
  # روی GPU قابل پردازش است
  # برای محاسبات عددی بهینه شده
X = torch.tensor(sequences, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)

# تقسیم داده به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(get_display(arabic_reshaper.reshape(f"داده آموزش: {len(X_train)} نمونه")))
print(get_display(arabic_reshaper.reshape(f"داده تست: {len(X_test)} نمونه")))

# ----------------------------
# ۷. ایجاد و آموزش مدل
# ----------------------------
# پارامترها
embedding_dim = 64 # یعنی هر کلمه با یک بردار 64 بعدی نمایش داده می‌شود  "گربه" -> [0.1, 0.5, -0.2, 0.8, ..., 0.3]  # 64 عدد
num_heads = 4      # Multi Head
num_classes = 3    # 0: منفی, 1: مثبت, 2: خنثی

model = SentimentAnalysisModel(vocab_size, embedding_dim, num_heads, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# آموزش مدل
num_epochs = 100
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    # حالت آموزش
    model.train()
    
    # Forward pass
    logits, _ = model(X_train)
    loss = criterion(logits, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # ارزیابی مدل
    model.eval()
    with torch.no_grad():
        test_logits, _ = model(X_test)
        test_preds = torch.argmax(test_logits, dim=1)
        test_accuracy = accuracy_score(y_test.numpy(), test_preds.numpy())
    
    train_losses.append(loss.item())
    test_accuracies.append(test_accuracy)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}')

# ----------------------------
# ۸. رسم نمودارها
# ----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title(get_display(arabic_reshaper.reshape('Loss طی دوره‌های آموزش')))
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(test_accuracies)
plt.title(get_display(arabic_reshaper.reshape('دقت تست طی دوره‌های آموزش')))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# ----------------------------
# ۹. تست مدل روی داده جدید
# ----------------------------
def predict_sentiment(text, model, vocab):
    # پیش‌پردازش متن
    processed_text = preprocess_text(text)
    sequence = text_to_sequence(processed_text, vocab, max_length)
    
    # پیش‌بینی
    # گرادیان مثل یک قطب نما است که به مدل نشان می‌دهد باید به کدام سمت حرکت کند تا خطا کم شود
    model.eval()
    with torch.no_grad():                                                # کاربرد: غیرفعال کردن محاسبه گرادیان‌ها
        input_tensor = torch.tensor([sequence], dtype=torch.long)        # چرا؟: در زمان تست نیازی به محاسبه gradient نداریم
        logits, attn_weights = model(input_tensor)                       # صرفه‌جویی در حافظه و محاسبات
        prediction = torch.argmax(logits, dim=1).item()
    
    # نمایش نتیجه
    sentiment_map = {0: 'منفی', 1: 'مثبت', 2:'خنثی'}
    return sentiment_map[prediction], attn_weights

# تست روی جملات جدید
test_texts = [
    "این محصول واقعا عالیه",
    "خیلی بد بود نخرید",
    "نه خوبه نه بد",
    "کیفیت خوبی داره",
    "مزخرف بود"
]

print(get_display(arabic_reshaper.reshape("\nنتایج پیش‌بینی روی داده جدید:")))
for text in test_texts:
    sentiment, _ = predict_sentiment(text, model, vocab)
    print(get_display(arabic_reshaper.reshape(f"'{text}' → احساس: {sentiment}")))

# ----------------------------
# ۱۰. نمایش وزن‌های attention
# ----------------------------
# گرفتن وزن‌های attention برای یک نمونه
model.eval()
with torch.no_grad():
    sample_input = X_test[0:1]
    _, attn_weights = model(sample_input)

# نمایش heatmap attention
plt.figure(figsize=(8, 6))
attn_matrix = attn_weights[0, 0].numpy()  # اولین head

plt.imshow(attn_matrix, cmap='hot', interpolation='nearest')
plt.title(get_display(arabic_reshaper.reshape('نقشه توجه (Attention Map)')))
plt.xlabel(get_display(arabic_reshaper.reshape('کلمات کلیدی')))
plt.ylabel(get_display(arabic_reshaper.reshape('کلمات پرسشی')))
plt.colorbar()

# اضافه کردن برچسب کلمات
words = []
for word_id in sample_input[0]:
    if word_id.item() != 0:  # ignore padding
        word = [k for k, v in vocab.items() if v == word_id.item()][0]
        words.append(get_display(arabic_reshaper.reshape(word)))

plt.xticks(range(len(words)), words, rotation=45)
plt.yticks(range(len(words)), words)

plt.tight_layout()
plt.show()

# ----------------------------
# ۱۱. گزارش طبقه‌بندی:
# ----------------------------
# تفسیر خروجی ها
  # Precision (دقت) : معنی: از بین پیش‌بینی‌های مثبت، چندتا واقعاً مثبت بودند
    # مثال: اگر مدل ۱۰ بار "مثبت" پیش‌بینی کند و ۸ تای آن درست باشد، precision = 0.8
  #  Recall (فراخوانی): معنی: از بین موارد واقعاً مثبت، چندتا را correctly شناسایی کرد؟
    # مثال: اگر ۱۰ نظر مثبت وجود داشته باشد و مدل ۹ تای آن را شناسایی کند، recall = 0.9
  # F1-Score: معنی: میانگین гарمونیک precision و recall
  # Support: معنی: تعداد نمونه‌های واقعی هر کلاس
  # Accuracy (دقت کلی) : معنی: درصد پیش‌بینی‌های درست از کل
  # Macro Avg: معنی: میانگین ساده معیارها برای همه کلاس‌ها
  # Weighted Avg: معنی: میانگین وزنی بر اساس support هر کلاس
    # برای کلاس‌های با نمونه بیشتر، وزن بیشتری دارد
print(get_display(arabic_reshaper.reshape("\nگزارش طبقه‌بندی:")))
model.eval()
with torch.no_grad():
    test_logits, _ = model(X_test)
    test_preds = torch.argmax(test_logits, dim=1)
    
    # بررسی کلاس‌های موجود
    unique_classes = np.unique(y_test.numpy())
    print(get_display(arabic_reshaper.reshape(f"کلاس‌های موجود: {unique_classes}")))
    
    # گزارش طبقه‌بندی
    print(classification_report(y_test.numpy(), test_preds.numpy(),
                              labels=[0, 1, 2],
                              target_names=[get_display(arabic_reshaper.reshape('منفی')), 
                                           get_display(arabic_reshaper.reshape('مثبت')), 
                                           get_display(arabic_reshaper.reshape('خنثی'))],
                              zero_division=0))  # برای جلوگیری از خطا در صورت عدم وجود کلاس