#                                                                         به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install numpy scikit-learn matplotlib pandas tabulate tensorflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

#----------------------------------------------------------------------------
# پروژه تحلیل احساسات با GRU
#----------------------------------------------------------------------------

#------------------------------
# 1- بارگذاری و آماده‌سازی داده‌ها
#------------------------------
# بارگذاری دیتاست IMDB (25000 نظر سینمایی)
vocab_size = 10000  # vocab_size=10000: فقط ۱۰۰۰۰ کلمه پرکاربرد را نگه می‌داریم.
max_len = 200       # max_len=200: اگر نظری بیش از ۲۰۰ کلمه داشت، قطع می‌شود و اگر کمتر بود، با صفر پر می‌شود.

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

# یکسان‌سازی طول دنباله‌ها با pad_sequences
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test =  tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

# نمایش شکل داده‌ها
print(get_display(arabic_reshaper.reshape("شکل داده‌های آموزشی:")), X_train.shape)  # (25000, 200)
print(get_display(arabic_reshaper.reshape("شکل داده‌های تست:")), X_test.shape)      # (25000, 200)

#------------------------------
# 2- ساخت مدل GRU با Regularization
#------------------------------

model = tf.keras.models.Sequential([
    # لایه تبدیل کلمات به بردارهای متراکم
      # input_dim : معنی: اندازه دایره‌واژگان (تعداد کلمات منحصر به فرد در داده‌های آموزشی)
      # output_dim : معنی: بعد بردار embedding (تعداد ویژگی‌های هر کلمه پس از تبدیل به بردار).
      # output_dim=128 یعنی هر کلمه به یک بردار ۱۲۸ بعدی تبدیل می‌شود
      # اعداد بزرگ‌تر → قدرت نمایش بهتر، اما محاسبات سنگین‌تر.
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    
    # لایه GRU با Dropout و BatchNorm
    tf.keras.layers.GRU(64, return_sequences=True),
    # Dropout: جلوگیری از بیش‌برازش با خاموش کردن تصادفی نورون‌ها.
    tf.keras.layers.Dropout(0.3),  # 30% از نورون‌ها در آموزش خاموش می‌شوند
    # BatchNorm: ثابت نگه داشتن مقیاس داده‌ها در لایه‌ها.
    tf.keras.layers.BatchNormalization(),
    
    # لایه GRU دوم
    tf.keras.layers.GRU(32),
    # Dropout: جلوگیری از بیش‌برازش با خاموش کردن تصادفی نورون‌ها.
    tf.keras.layers.Dropout(0.3),
    # BatchNorm: ثابت نگه داشتن مقیاس داده‌ها در لایه‌ها.
    tf.keras.layers.BatchNormalization(),
    
    # لایه خروجی
    tf.keras.layers.Dense(1, activation='sigmoid')  # خروجی بین 0 (منفی) و 1 (مثبت)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

#------------------------------
# 3- آموزش مدل با EarlyStopping
#------------------------------

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,  # اگر در ۳ دوره val_loss بهبود نیافت، آموزش متوقف شود
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,  # 20% داده آموزش برای اعتبارسنجی
    callbacks=[early_stop],
    verbose=1
)

#------------------------------
# 4- ارزیابی مدل روی داده تست
#------------------------------

# محاسبه دقت و خطا
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(get_display(arabic_reshaper.reshape(f'\nدقت نهایی روی داده تست: {test_acc*100:.2f}%')))

# پیش‌بینی‌ها
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)  # تبدیل احتمالات به کلاس 0/1

# محاسبه MSE و MAE
print(f'MSE: {mean_squared_error(y_test, y_pred):.4f}')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}')

# نمایش ۵ نمونه تصادفی
indices = np.random.choice(len(X_test), 5)
for i in indices:
    print(get_display(arabic_reshaper.reshape(f'\nنظر واقعی: {"مثبت" if y_test[i]==1 else "منفی"} | پیش‌بینی مدل: {"مثبت" if y_pred_classes[i]==1 else "منفی"} ({y_pred[i][0]:.2f})')))

#------------------------------
# 5- رسم نمودارهای آموزش
#------------------------------

plt.figure(figsize=(12, 5))

# نمودار دقت
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label=get_display(arabic_reshaper.reshape('آموزش')))
plt.plot(history.history['val_accuracy'], label=get_display(arabic_reshaper.reshape('اعتبارسنجی')))
plt.title(get_display(arabic_reshaper.reshape('نمودار دقت')))
plt.xlabel(get_display(arabic_reshaper.reshape('دوره')))
plt.ylabel(get_display(arabic_reshaper.reshape('دقت')))
plt.legend()

# نمودار خطا
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label=get_display(arabic_reshaper.reshape('آموزش')))
plt.plot(history.history['val_loss'], label=get_display(arabic_reshaper.reshape('اعتبارسنجی')))
plt.title(get_display(arabic_reshaper.reshape('نمودار خطا')))
plt.xlabel(get_display(arabic_reshaper.reshape('دوره')))
plt.ylabel(get_display(arabic_reshaper.reshape('خطا')))
plt.legend()

plt.tight_layout()
plt.show()