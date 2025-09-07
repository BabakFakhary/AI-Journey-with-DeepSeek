#                                                                         به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import pickle
import json
import matplotlib.pyplot as plt

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

# ==================================================================================================
#  پیاده‌سازی Transformer با TensorFlow/Keras
# ==================================================================================================

# ایجاد دیتاست متن نمونه
texts = [
    "I love this movie it is great",
    "Fantastic film with amazing acting",
    "Wonderful story and beautiful cinematography",
    "Terrible waste of time boring",
    "Horrible acting and bad plot",
    "Awful experience would not recommend",
    "This is fantastic brilliant",
    "Excellent performance by all actors",
    "Poor direction and weak script",
    "Amazing visuals and great soundtrack"
]

labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 1]  # 1: مثبت, 0: منفی

# تبدیل به DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})
print("Dataset:")
print(df.head())

# --------------------------------------------------------------------------
# ساخت مدل Transformer با TensorFlow/Keras
# --------------------------------------------------------------------------

# لایه Positional Encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Extract the required parameters
        position = config.get('position')
        d_model = config.get('d_model')
        
        # Remove these from config so they're not passed twice
        config_copy = config.copy()
        config_copy.pop('position', None)
        config_copy.pop('d_model', None)
        
        return cls(position, d_model, **config_copy)

# لایه Transformer Block
def transformer_block(d_model, num_heads, d_ff, rate=0.1):
    inputs = tf.keras.layers.Input(shape=(None, d_model))
    
    # Multi-Head Attention
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model//num_heads)(inputs, inputs)
    attention = tf.keras.layers.Dropout(rate)(attention)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    
    # Feed Forward Network
    ffn = tf.keras.layers.Dense(d_ff, activation='relu')(out1)
    ffn = tf.keras.layers.Dense(d_model)(ffn)
    ffn = tf.keras.layers.Dropout(rate)(ffn)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)
    
    return tf.keras.Model(inputs, out2)

# --------------------------------------------------------------------------
# ساخت مدل کامل
# --------------------------------------------------------------------------
def build_transformer_model(vocab_size, d_model, num_heads, d_ff, num_layers, max_length, num_classes):
    inputs = tf.keras.layers.Input(shape=(max_length,))
    
    # Embedding
    embedding = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    
    # Positional Encoding
    positional_encoding = PositionalEncoding(max_length, d_model)
    encoded = positional_encoding(embedding)
    
    # Transformer Blocks
    for _ in range(num_layers):
        transformer_layer = transformer_block(d_model, num_heads, d_ff)
        encoded = transformer_layer(encoded)
    
    # Global Average Pooling
    pooled = tf.keras.layers.GlobalAveragePooling1D()(encoded)
    
    # Classification Head
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(pooled)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# پارامترهای مدل
VOCAB_SIZE = 1000
D_MODEL = 128
NUM_HEADS = 4
D_FF = 512
NUM_LAYERS = 2
MAX_LENGTH = 20  # کاهش طول به دلیل داده‌های کوچک
NUM_CLASSES = 2

# ساخت مدل
model = build_transformer_model(
    VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_LENGTH, NUM_CLASSES
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("Model summary:")
model.summary()

# --------------------------------------------------------------------------
#  Training و Evaluation
# --------------------------------------------------------------------------

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)

# تبدیل متون به sequences
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=MAX_LENGTH, padding='post', truncating='post'
)

print("Sample sequence:", sequences[0])
print("Padded sequence shape:", padded_sequences.shape)

# تقسیم داده
X_seq_train, X_seq_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.3, random_state=42, stratify=labels
)

# تبدیل داده‌ها به numpy arrays با نوع داده مناسب
X_seq_train = np.array(X_seq_train, dtype=np.int32)
X_seq_test = np.array(X_seq_test, dtype=np.int32)
y_train = np.array(y_train, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)

print(f"Train sequences: {X_seq_train.shape}, Test sequences: {X_seq_test.shape}")

# آموزش مدل
history = model.fit(
    X_seq_train, y_train,
    batch_size=4,  # کاهش batch size
    epochs=30,     # کاهش epochs
    validation_data=(X_seq_test, y_test),
    verbose=1
)

# ذخیره تاریخچه آموزش
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# ارزیابی مدل
test_loss, test_acc = model.evaluate(X_seq_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# پیش‌بینی روی داده تست
y_pred = model.predict(X_seq_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# گزارش classification
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=['Negative', 'Positive'], zero_division=0))

# --------------------------------------------------------------------------
# Fine-tuning با داده جدید 
# --------------------------------------------------------------------------

# داده جدید برای fine-tuning
new_texts = [
    "Absolutely brilliant performance",
    "Waste of money and time",
    "Outstanding cinematography",
    "Disappointing ending",
    "Masterpiece of filmmaking"
]

new_labels = [1, 0, 1, 0, 1]

# پردازش متن جدید
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = tf.keras.preprocessing.sequence.pad_sequences(
    new_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post'
)

# تبدیل به numpy arrays
new_padded = np.array(new_padded, dtype=np.int32)
new_labels = np.array(new_labels, dtype=np.int32)

print("New data processed:")
for i, text in enumerate(new_texts):
    print(f"Text: {text} -> Label: {new_labels[i]}")

# Fine-tuning مدل با learning rate پایین‌تر
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy()
)

# Fine-tuning روی داده جدید
fine_tune_history = model.fit(
    new_padded, new_labels,
    batch_size=2,
    epochs=10,
    verbose=1
)

print("Fine-tuning completed!")

# ارزیابی پس از Fine-tuning
test_loss_ft, test_acc_ft = model.evaluate(X_seq_test, y_test, verbose=0)
print(f"After Fine-tuning - Test Loss: {test_loss_ft:.4f}")
print(f"After Fine-tuning - Test Accuracy: {test_acc_ft:.4f}")

# مقایسه با قبل از fine-tuning
print(f"\nComparison:")
print(f"Before FT - Accuracy: {test_acc:.4f}")
print(f"After FT - Accuracy: {test_acc_ft:.4f}")
print(f"Improvement: {test_acc_ft - test_acc:.4f}")

# --------------------------------------------------------------------------
# ذخیره و بارگذاری مدل
# --------------------------------------------------------------------------

# ذخیره مدل با فرمت Keras (جدید)
model.save('transformer_sentiment_model.keras')
print("Model saved successfully!")

# ذخیره توکنایزر
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ذخیره پارامترهای مدل
model_config = {
    'vocab_size': VOCAB_SIZE,
    'd_model': D_MODEL,
    'num_heads': NUM_HEADS,
    'd_ff': D_FF,
    'num_layers': NUM_LAYERS,
    'max_length': MAX_LENGTH,
    'num_classes': NUM_CLASSES
}

with open('model_config.json', 'w') as f:
    json.dump(model_config, f)

print("Tokenizer and config saved!")

# --------------------------------------------------------------------------
# بارگذاری مدل
# --------------------------------------------------------------------------

# بارگذاری مدل
loaded_model = tf.keras.models.load_model('transformer_sentiment_model.keras', 
                                         custom_objects={'PositionalEncoding': PositionalEncoding})

# بارگذاری توکنایزر
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# بارگذاری config
with open('model_config.json', 'r') as f:
    loaded_config = json.load(f)

print("Model and tokenizer loaded successfully!")

# --------------------------------------------------------------------------
# تست مدل بارگذاری شده
# --------------------------------------------------------------------------

test_text = "This movie is absolutely wonderful"
test_sequence = loaded_tokenizer.texts_to_sequences([test_text])
test_padded = tf.keras.preprocessing.sequence.pad_sequences(
    test_sequence, maxlen=MAX_LENGTH, padding='post', truncating='post'
)

# تبدیل به numpy array
test_padded = np.array(test_padded, dtype=np.int32)

prediction = loaded_model.predict(test_padded)
sentiment = "Positive" if np.argmax(prediction) == 1 else "Negative"
confidence = np.max(prediction)

print(f"Text: '{test_text}'")
print(f"Sentiment: {sentiment} (Confidence: {confidence:.3f})")

print(fa("\n🎉 پروژه کامل شد!"))

# --------------------------------------------------------------------------
# تجسم نتایج
# --------------------------------------------------------------------------

# تجسم تاریخچه آموزش
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()