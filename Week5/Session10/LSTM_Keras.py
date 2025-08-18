#                                                                                     به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install numpy scikit-learn matplotlib pandas tabulate tensorflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# مثال سری زمانی سینوسی با نویز
# -----------------------------------------------------------------------------

# تولید داده مصنوعی (سینوسی با نویز)
def generate_data(length=1000):
    x = np.linspace(0, 20, length)
    y = np.sin(x) + np.random.normal(0, 0.1, length)
    return y

data = generate_data()

# تبدیل به فرمت مناسب LSTM
def create_dataset(data, n_steps=10):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

# با n_steps=10، مدل از 10 نقطه قبلی برای پیش‌بینی نقطه بعدی استفاده می‌کند
n_steps = 10
X, y = create_dataset(data, n_steps)

# تقسیم داده به آموزش و تست (80-20)
# shuffle=False برای حفظ ترتیب زمانی.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# تغییر شکل برای LSTM (نمونه‌ها, گام‌های زمانی, ویژگی‌ها)
X_train = X_train.reshape(-1, n_steps, 1)
X_test = X_test.reshape(-1, n_steps, 1)

# ساخت مدل LSTM
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='tanh', input_shape=(n_steps, 1)),
    tf.keras.layers.Dense(1)  # خروجی پیوسته
])

model.compile(optimizer='adam', 
              loss='mse',  # MSE برای رگرسیون
              metrics=['mae'])  # MAE برای ارزیابی

# آموزش مدل
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    verbose=1)

#  ارزیابی مدل روی داده تست

# پیش‌بینی و محاسبه خطاها
y_pred = model.predict(X_test)

# MSE (خطای مربعات میانگین): برای ارزیابی کلی مدل.
mse = mean_squared_error(y_test, y_pred)
# MAE (خطای مطلق میانگین): برای تفسیرپذیری بهتر.
mae = mean_absolute_error(y_test, y_pred)
print(f'MSE: {mse:.4f}, MAE: {mae:.4f}')

# نمایش ۵ نمونه تصادفی
for i in np.random.randint(0, len(y_test), 5):
    print(get_display(arabic_reshaper.reshape(f'واقعی: {y_test[i]:.2f} | پیش‌بینی: {y_pred[i][0]:.2f}')))

# بهینه‌سازی:
# افزایش n_steps برای وابستگی‌های طولانی‌تر.
# تنظیم تعداد نورون‌های LSTM برای دقت بهتر.

# رسم نمودارها
plt.figure(figsize=(15, 5))

# نمودار پیش‌بینی vs واقعی
plt.subplot(1, 2, 1)
plt.plot(y_test, label=get_display(arabic_reshaper.reshape('مقادیر واقعی')), alpha=0.7)
plt.plot(y_pred, label=get_display(arabic_reshaper.reshape('پیش‌بینی‌ها')), alpha=0.7)
plt.title(get_display(arabic_reshaper.reshape('مقایسه پیش‌بینی و واقعیت')))
plt.legend()

# نمودارهای آموزش
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label=get_display(arabic_reshaper.reshape('آموزش')))
plt.plot(history.history['val_loss'], label=get_display(arabic_reshaper.reshape('تست')))
plt.title(get_display(arabic_reshaper.reshape('نمودار خطا (MSE)')))
plt.legend()

plt.tight_layout()
plt.show()