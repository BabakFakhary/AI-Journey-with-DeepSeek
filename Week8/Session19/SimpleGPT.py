#                                                                               به نام خدا
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPTAttention(nn.Module):
    """لایه Attention برای GPT (با masking)"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # لایه‌های linear برای Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # لایه خروجی
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # تولید Q, K, V
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # تغییر شکل برای multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # محاسبه attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # اعمال mask برای جلوگیری از نگاه به آینده
        if mask is not None:
            # mask با شکل (batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # اعمال softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # ضرب در values
        context = torch.matmul(attention_weights, V)
        
        # تغییر شکل به حالت اصلی
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_linear(context), attention_weights

class GPTBlock(nn.Module):
    """یک بلوک کامل GPT"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Self-Attention
        self.attention = GPTAttention(d_model, num_heads)
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Self-Attention با residual connection
        attn_output, attn_weights = self.attention(x, mask)
        attn_output = self.attention_dropout(attn_output)
        x = self.attention_norm(x + attn_output)
        
        # Feed Forward با residual connection
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        
        return x, attn_weights

class SimpleGPT(nn.Module):
    """یک GPT ساده"""
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, d_ff=3072, max_seq_len=512):
        super().__init__()
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # GPT blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # Layer norm final
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output layer
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (توکن embedding و output sharing)
        self.lm_head.weight = self.token_embedding.weight
        
        # ایجاد causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # ایجاد embeddings
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)
        
        x = token_embeds + position_embeds
        
        # ایجاد mask
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # عبور از بلوک‌ها
        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, causal_mask)
            attention_weights.append(attn_weights)
        
        # نرمال‌سازی نهایی
        x = self.final_norm(x)
        
        # محاسبه logits
        logits = self.lm_head(x)
        
        return logits, attention_weights

# تست مدل ساخته شده
vocab_size = 50257  # GPT-2 vocabulary size
model = SimpleGPT(vocab_size=vocab_size, d_model=768, num_heads=12, num_layers=12)

# ایجاد یک ورودی نمونه
input_ids = torch.randint(0, vocab_size, (2, 32))  # batch_size=2, seq_len=32

# Forward pass
logits, attention_weights = model(input_ids)
print(f"Input shape: {input_ids.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Number of attention layers: {len(attention_weights)}")