#                                                                         به نام خدا
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
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------------------------------------------------------------
# در این مثال، از یک دنباله ساده عددی استفاده می‌کنیم تا مدل یاد بگیرد الگوها را تشخیص دهد و اعداد بعدی را پیش‌بینی کند.
# (مثلاً دنباله: 0.1, 0.2, 0.3, ... → مدل باید عدد بعدی را حدس بزند)
# ----------------------------------------------------------------------------------------------------------------------------

# تولید دنباله سینوسی با نویز
def generate_sequence(length=100):
    # np.linspace(start, stop, num=50, endpoint=True) start: عدد شروع stop: عدد پایان num: تعداد نقاط (پیش‌فرض: ۵۰)
    # numbers = np.linspace(0, 1, 5) [0.   0.25 0.5  0.75 1.  ]
    x = np.linspace(0, 10, length)
    y = np.sin(x) + np.random.normal(0, 0.1, size=len(x))  # سینوس با نویز
    return y

data = generate_sequence()

# تقسیم داده به آموزشی و تست
train = data[:80] # تمام عناصر از شروع آرایه تا عنصر 80
test = data[80:]  # تمام عناصر از عنصر 80 تا پایان آرایه

# RNN به داده با شکل (نمونه‌ها, گام‌های زمانی, ویژگی‌ها) نیاز دارد.
# تبدیل داده به فرمت مناسب RNN
def prepare_data(sequence, n_steps=3):
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        X.append(sequence[i:i+n_steps])
        y.append(sequence[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 3
X_train, y_train = prepare_data(train, n_steps)
X_test, y_test = prepare_data(test, n_steps)

# تغییر شکل داده برای RNN (نمونه‌ها, گام‌های زمانی, ویژگی‌ها)
# ------------------------Resharp------------------------------
# X_train.reshape(-1, 1)    
# 1: تعداد ستون‌ها را ۱ تنظیم می‌کند.
# -1: به NumPy می‌گوید: "تعداد سطرها را خودت محاسبه کن" (با حفظ تعداد کل عناصر). 
# ------------------------------------------------------------
# X_train = np.array([0.1, 0.2, 0.3, ..., 10.0])
# X_train = X_train.reshape(-1, 5 , 1)  # شکل جدید: (20, 5, 1)
# اگر n_steps=5 (یعنی هر دنباله ۵ عدد دارد):
# [[0.1], [0.2], [0.3], [0.4], [0.5]],  # دنباله اول
# [[0.6], [0.7], [0.8], [0.9], [1.0]],  # دنباله دوم
# ------------------------------------------------------------
# تغییر شکل داده برای RNN (نمونه‌ها, گام‌های زمانی, ویژگی‌ها)
X_train = X_train.reshape(-1, n_steps, 1)
X_test = X_test.reshape(-1, n_steps, 1)

# ساخت مدل SimpleRNN
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(50,                         # تعداد نورون‌ها
                              activation='tanh',          # تابع فعال‌سازی (معمولاً tanh یا relu)
                              input_shape=(n_steps, 1)),  # شکل ورودی
    tf.keras.layers.Dense(1)  # خروجی پیوسته (پیش‌بینی عدد)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# آموزش مدل
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=0
)

# پیش‌بینی روی داده تست
y_pred = model.predict(X_test)

# محاسبه خطا
mse = mean_squared_error(y_test, y_pred)
print(get_display(arabic_reshaper.reshape(f"خطای مربعات میانگین (MSE): {mse:.4f}")))

# نمایش 5 پیش‌بینی نمونه
for i in range(5):
    print(get_display(arabic_reshaper.reshape(f"واقعی: {y_test[i]:.2f} | پیش‌بینی: {y_pred[i][0]:.2f}")))
