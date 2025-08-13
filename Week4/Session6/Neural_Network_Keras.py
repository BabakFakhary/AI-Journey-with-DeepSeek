#                                                                                  به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install numpy scikit-learn matplotlib pandas tabulate tensorflow
import tensorflow as tf
import numpy as np

# تولید داده‌های مصنوعی (مثال)
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# تقسیم داده به Train/Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تعریف مدل
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),  # لایه پنهان با 4 نورون
    tf.keras.layers.Dense(1, activation='sigmoid')                 # لایه خروجی
])

# کامپایل مدل
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# آموزش مدل
# epochs یک پاس کامل روی تمام داده‌های آموزشی
# verbose کنترل می‌کند که اطلاعات آموزش چگونه نمایش داده شوند 
model.fit(X_train, y_train, epochs=100, verbose=0)

# پیش‌بینی برای داده‌های تست
y_pred = model.predict(X_test)
#  برای اینکه خروجی واقعی یک یا صفر هست عدی که پیش بینی می کند عددی بین صفر و یک هست باید برای این کار به صفر یا یک تبدیل کنیم  
y_pred_classes = (y_pred > 0.5).astype(int).flatten()

# محاسبه خطا برای هر نمونه
for i in range(len(X_test)):
    sample = X_test[i]
    true_label = y_test[i]
    predicted_prob = y_pred[i][0]
    predicted_label = y_pred_classes[i]
    
    # محاسبه خطا برای این نمونه
    error = abs(true_label - predicted_prob)
    
    print(get_display(arabic_reshaper.reshape(f"نمونه {i+1}:")))
    print(get_display(arabic_reshaper.reshape(f"  ویژگی‌ها: {sample}")))
    print(get_display(arabic_reshaper.reshape(f"  برچسب واقعی: {true_label}")))
    print(get_display(arabic_reshaper.reshape(f"  پیش‌بینی مدل (احتمال): {predicted_prob:.4f}")))
    print(get_display(arabic_reshaper.reshape(f"  پیش‌بینی نهایی (کلاس): {predicted_label}")))
    print(get_display(arabic_reshaper.reshape(f"  خطای این نمونه: {error:.4f}")))
    print("-" * 40)

# محاسبه و نمایش نمونه‌های اشتباه
wrong_predictions = np.where(y_pred_classes != y_test)[0]
print(get_display(arabic_reshaper.reshape(f"\nتعداد نمونه‌های اشتباه: {len(wrong_predictions)} از {len(y_test)}")))
print(get_display(arabic_reshaper.reshape("نمونه‌های اشتباه:")))
for idx in wrong_predictions:
    print(get_display(arabic_reshaper.reshape(f"  نمونه {idx+1}: واقعی={y_test[idx]}, پیش‌بینی={y_pred_classes[idx]}")))

# ارزیابی مدل
loss, accuracy = model.evaluate(X_test, y_test)
print(get_display(arabic_reshaper.reshape(f"دقت مدل روی داده تست: {accuracy * 100:.2f}%")))
print(get_display(arabic_reshaper.reshape(f"خطای کلی: {loss:.4f}\n")))