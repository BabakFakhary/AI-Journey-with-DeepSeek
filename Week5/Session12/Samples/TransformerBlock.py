#                                                                          به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

# --------------------------------------------------------------------------------------------------------
# Transformer Block
# Input → Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm → Output
# Multi-Head Attention: درک روابط بین کلمات
# Feed Forward Network (FFN): پردازش غیرخطی
# Residual Connection: جلوگیری از vanishing gradient
# Layer Normalization: پایدارسازی آموزش
# --------------------------------------------------------------------------------------------------------

# تنظیمات نمایش فارسی
def fa(text):
  return get_display(arabic_reshaper.reshape(text))

# =============================================================================
# تعریف تابع residual_connection
# =============================================================================
def residual_connection(x, sublayer_output):
    return x + sublayer_output

# =============================================================================
# تعریف کلاس LayerNormalization
# =============================================================================
class LayerNormalization:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones((1, 1, d_model))
        self.beta = np.zeros((1, 1, d_model))
        self.eps = eps
        
    def forward(self, x):
        # محاسبه میانگین و واریانس
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # نرمال‌سازی
        x_normalized = (x - mean) / np.sqrt(variance + self.eps)
        
        # مقیاس و انتقال
        return self.gamma * x_normalized + self.beta
    
# تابع tokenization ساده
def simple_tokenizer(text, vocab):
    tokens = text.lower().split()
    return [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]

# ایجاد vocabulary ساده برای تست
vocab = {'the': 1, 'cat': 2, 'chased': 3, 'mouse': 4, 'dog': 5, 'runs': 6, 
         'sleeps': 7, 'eats': 8, 'cheese': 9, 'bird': 10, 'sings': 11, 
         'beautifully': 12, 'a': 13, '<UNK>': 0}

reverse_vocab = {v: k for k, v in vocab.items()}

# =============================================================================
# تعریف کلاس MultiHeadAttention
# =============================================================================
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        # ماتریس‌های وزن
        self.WQ = np.random.randn(d_model, d_model) * 0.01
        self.WK = np.random.randn(d_model, d_model) * 0.01
        self.WV = np.random.randn(d_model, d_model) * 0.01
        self.WO = np.random.randn(d_model, d_model) * 0.01
        
    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        matmul_qk = np.matmul(Q, K.transpose(0, 1, 3, 2))
        dk = K.shape[-1]
        scaled_attention_logits = matmul_qk / np.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = self.softmax(scaled_attention_logits, axis=-1)
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def call(self, x, mask=None):
        batch_size = x.shape[0]
        
        # تولید Q, K, V
        Q = np.dot(x, self.WQ)
        K = np.dot(x, self.WK)
        V = np.dot(x, self.WV)
        
        # تقسیم به heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # محاسبه attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # ترکیب heads
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        
        # لایه خروجی
        output = np.dot(concat_attention, self.WO)
        
        return output, attention_weights

# =============================================================================
# تعریف کلاس FeedForwardNetwork
# =============================================================================
class FeedForwardNetwork:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # ماتریس‌های وزن
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((1, 1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((1, 1, d_model))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def call(self, x):
        # لایه اول
        hidden = np.dot(x, self.W1) + self.b1
        hidden = self.relu(hidden)
        
        # لایه دوم
        output = np.dot(hidden, self.W2) + self.b2
        
        return output

# =============================================================================
# تعریف کلاس TransformerBlock
# =============================================================================
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.ln1 = LayerNormalization(d_model)
        self.ln2 = LayerNormalization(d_model)
        
    def call(self, x, mask=None):
        # Multi-Head Attention
        attn_output, attn_weights = self.mha.call(x, mask)
        
        # Residual Connection + Layer Norm
        x1 = residual_connection(x, attn_output)
        x1 = self.ln1.forward(x1)
        
        # Feed Forward Network
        ffn_output = self.ffn.call(x1)
        
        # Residual Connection + Layer Norm
        x2 = residual_connection(x1, ffn_output)
        output = self.ln2.forward(x2)
        
        return output, attn_weights

# =============================================================================
# تست ابتدایی
# =============================================================================
print(fa("🔧 شروع تست‌های اولیه..."))

# تست Multi-Head Attention
d_model = 64
num_heads = 4
batch_size = 1
seq_len = 5

mha = MultiHeadAttention(d_model, num_heads)
x = np.random.randn(batch_size, seq_len, d_model)

output, attention_weights = mha.call(x)
print("\n✅ Multi-Head Attention Tested")
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", attention_weights.shape)

# تست FFN
d_ff = 256
ffn = FeedForwardNetwork(d_model, d_ff)
x = np.random.randn(batch_size, seq_len, d_model)
output = ffn.call(x)
print("\n✅ FeedForwardNetwork Tested")
print("Input shape:", x.shape)
print("Output shape:", output.shape)

# تست Residual Connection
original = np.random.randn(2, 3, 4)
sublayer_output = np.random.randn(2, 3, 4) * 0.1
result = residual_connection(original, sublayer_output)
print("\n✅ Residual Connection Tested")
print("Original shape:", original.shape)
print("Sublayer output shape:", sublayer_output.shape)
print("Result shape:", result.shape)

# تست Transformer Block
transformer_block = TransformerBlock(d_model, num_heads, d_ff)
x = np.random.randn(batch_size, seq_len, d_model)
output, attn_weights = transformer_block.call(x)
print("\n✅ Transformer Block Tested")
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", attn_weights.shape)

# =============================================================================
# ایجاد embedding layer و positional encoding ساده
# =============================================================================
vocab_size = len(vocab)
embedding_dim = d_model

# ایجاد embedding layer ساده
embedding_layer = np.random.randn(vocab_size, embedding_dim) * 0.1

# ایجاد positional encoding ساده
max_seq_len = 20
pe = np.zeros((max_seq_len, embedding_dim))
for pos in range(max_seq_len):
    for i in range(0, embedding_dim, 2):
        pe[pos, i] = np.sin(pos / (10000 ** (2 * i / embedding_dim)))
        if i + 1 < embedding_dim:
            pe[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / embedding_dim)))

# =============================================================================
# پردازش جمله واقعی
# =============================================================================
def process_sentence_with_transformer(sentence, vocab, embedding_layer, pe, transformer_block):
    # Tokenization و Embedding
    tokens = simple_tokenizer(sentence, vocab)
    if not tokens:
        return None
        
    word_embeddings = embedding_layer[tokens]
    
    # اضافه کردن Positional Encoding
    position_ids = np.arange(len(tokens))
    positional_embeddings = pe[position_ids]
    combined_embeddings = word_embeddings + positional_embeddings
    
    # تبدیل به batch
    batch_input = combined_embeddings[np.newaxis, :, :]
    
    # پردازش با Transformer Block
    output, attention_weights = transformer_block.call(batch_input)
    
    return {
        'tokens': tokens,
        'word_embeddings': word_embeddings,
        'output': output[0],  # حذف بعد batch
        'attention_weights': attention_weights[0]  # حذف بعد batch
    }

# =============================================================================
# تجسم Attention Weights
# =============================================================================
def visualize_attention(attention_weights, words, head_idx=0):
    """تجسم وزن‌های attention برای یک head"""
    if len(attention_weights) <= head_idx:
        print(fa(f"Head {head_idx} وجود ندارد"))
        return
        
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights[head_idx], cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f'Attention Weights - Head {head_idx+1}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.xticks(range(len(words)), words, rotation=45)
    plt.yticks(range(len(words)), words)
    plt.tight_layout()
    plt.show()

# =============================================================================
# تحلیل تغییرات پس از Transformer Block
# =============================================================================
def analyze_transformation(original, transformed, words):
    """تحلیل تغییرات پس از Transformer Block"""
    print(fa("📊 تحلیل تغییرات:"))
    print("=" * 50)
    
    # محاسبه تغییرات
    changes = np.abs(transformed - original)
    
    print(fa(f"میانگین تغییرات: {np.mean(changes):.6f}"))
    print(fa(f"ماکزیمم تغییرات: {np.max(changes):.6f}"))
    print(fa(f"مینیمم تغییرات: {np.min(changes):.6f}"))
    
    # تغییرات per token
    print(fa("\nتغییرات per token:"))
    for i, word in enumerate(words):
        token_change = np.mean(changes[i])
        print(f"  {word}: {token_change:.6f}")

# =============================================================================
# بررسی شباهت‌های语义
# =============================================================================
def analyze_semantic_similarity(embeddings, words):
    """بررسی شباهت‌های بین کلمات"""
    print(fa("\n🔍 تحلیل شباهت‌های:"))
    print("=" * 50)
    
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words[i+1:], i+1):
            # محاسبه cosine similarity
            vec1 = embeddings[i]
            vec2 = embeddings[j]
            similarity = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )
            print(f"  {word1} - {word2}: {similarity:.3f}")

# =============================================================================
# پردازش جمله نمونه اصلی
# =============================================================================
print("\n" + "="*60)
print(fa("پردازش جمله نمونه اصلی"))
print("="*60)

sentence = "The cat chased the mouse"
results = process_sentence_with_transformer(
    sentence, vocab, embedding_layer, pe, transformer_block
)

if results:
    print(fa("جمله:"), sentence)
    print("Tokens:", results['tokens'])
    words = [reverse_vocab.get(t, 'UNK') for t in results['tokens']]
    print("Words:", words)
    print("Output shape:", results['output'].shape)
    
    # تجسم برای همه headها
    for i in range(min(num_heads, len(results['attention_weights']))):
        visualize_attention(results['attention_weights'], words, i)
    
    # اجرای تحلیل
    original_embeddings = results['word_embeddings'] + pe[:len(results['tokens'])]
    transformed_embeddings = results['output']
    
    analyze_transformation(original_embeddings, transformed_embeddings, words)
    
    print(fa("قبل از Transformer Block:"))
    analyze_semantic_similarity(original_embeddings, words)
    
    print(fa("\nبعد از Transformer Block:"))
    analyze_semantic_similarity(transformed_embeddings, words)
else:
    print(fa("خطا در پردازش جمله"))

# =============================================================================
# تست با جملات مختلف
# =============================================================================
print("\n" + "="*60)
print(fa("تست با جملات مختلف"))
print("="*60)

test_sentences = [
    "The cat sleeps",
    "The dog runs", 
    "The mouse eats cheese",
    "A bird sings beautifully"
]

for test_sentence in test_sentences:
    print(f"\n{'='*60}")
    print(fa(f"پردازش جمله: '{test_sentence}'"))
    print(f"{'='*60}")
    
    try:
        # پردازش جمله
        results = process_sentence_with_transformer(
            test_sentence, vocab, embedding_layer, pe, transformer_block
        )
        
        if results:
            words = [reverse_vocab.get(t, 'UNK') for t in results['tokens']]
            print(fa(f"کلمات: {words}"))
            print(fa(f"خروجی shape: {results['output'].shape}"))
            
            # بررسی تغییرات
            original = results['word_embeddings'] + pe[:len(results['tokens'])]
            transformed = results['output']
            avg_change = np.mean(np.abs(transformed - original))
            print(fa(f"میانگین تغییرات: {avg_change:.6f}"))
        else:
            print(fa("جمله پردازش نشد"))
            
    except Exception as e:
        print(f"خطا: {e}")

print(fa("\n🎉 تمام تست‌ها با موفقیت انجام شد!"))