#                                                                        به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install numpy scikit-learn matplotlib pandas tabulate tensorflow
# کتابخانه‌های مورد نیاز
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------------------------------------------------------------
# پیش بینی عدد از روی تصویر
# -------------------------------------------------------------------


# 1. بارگذاری و پیش‌پردازش داده‌ها
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# نرمال‌سازی و تغییر شکل داده‌ها
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. ساخت مدل CNN
model =  tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
    filters=32,         # تعداد فیلترها (تعداد ویژگی‌های استخراج شده مانند بافت لبه و ...)
    kernel_size=(3, 3),  # ابعاد پنجره کانولوشن (height, width)
    activation='relu',   # تابع فعال‌سازی (غیرخطی‌سازی) relu (Rectified Linear Unit) یک تابع فعال‌سازی غیرخطی است که به مدل کمک می‌کند روابط پیچیده را یاد بگیرد.
    input_shape=(28, 28, 1)  # ابعاد ورودی (height, width, channels)
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. آموزش مدل
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=64,
                    validation_split=0.2)

# 4. ارزیابی روی داده تست
test_loss, test_acc = model.evaluate(X_test, y_test)
print(get_display(arabic_reshaper.reshape(f"\nدقت نهایی روی داده تست: {test_acc*100:.2f}%")))

# 5. پیش‌بینی روی نمونه‌های تست (5 نمونه اول)
predictions = model.predict(X_test[:5])
# اندیس (موقعیت) بزرگ‌ترین مقدار را در یک آرایه در هر ستون از سطر پیدا می‌کند. 
# در فایل ورد توضیح داده شده
predicted_labels = np.argmax(predictions, axis=1) 

print(get_display(arabic_reshaper.reshape("\nپیش‌بینی برای 5 نمونه اول تست:")))
for i in range(5):
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(get_display(arabic_reshaper.reshape(f"پیش‌بینی: {predicted_labels[i]}, واقعی: {np.argmax(y_test[i])}")))
    plt.show()

# 6. رسم نمودارهای آموزش
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label=get_display(arabic_reshaper.reshape('دقت آموزش')))
plt.plot(history.history['val_accuracy'], label=get_display(arabic_reshaper.reshape('دقت اعتبارسنجی')))
plt.title(get_display(arabic_reshaper.reshape('نمودار دقت')))
plt.xlabel(get_display(arabic_reshaper.reshape('دوره‌های آموزش')))
plt.ylabel(get_display(arabic_reshaper.reshape('دقت')))
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label=get_display(arabic_reshaper.reshape('خطای آموزش')))
plt.plot(history.history['val_loss'], label=get_display(arabic_reshaper.reshape('خطای اعتبارسنجی')))
plt.title(get_display(arabic_reshaper.reshape('نمودار خطا')))
plt.xlabel(get_display(arabic_reshaper.reshape('دوره‌های آموزش')))
plt.ylabel(get_display(arabic_reshaper.reshape('خطا')))
plt.legend()

plt.tight_layout()
plt.show()