#                                                                                 به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install numpy scikit-learn matplotlib pandas tabulate tensorflow
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# داده‌های ویژگی (X) و برچسب (y) - کاملاً قابل فهم
# ویژگی‌ها: [سن, درآمد] (سن به سال، درآمد به میلیون تومان)
X = np.array([
    [25, 30],  # نمونه 1
    [30, 40],  # نمونه 2
    [35, 35],  # نمونه 3
    [40, 50],  # نمونه 4
    [20, 20],  # نمونه 5
    [50, 60],  # نمونه 6
    [45, 45],  # نمونه 7
    [55, 30]   # نمونه 8
])

# برچسب‌ها: 0 = "غیرخریدار"، 1 = "خریدار"
y = np.array([0, 1, 1, 1, 0, 1, 1, 0])

# تقسیم داده به Train و Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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

# ارزیابی مدل
loss, accuracy = model.evaluate(X_test, y_test)
print(get_display(arabic_reshaper.reshape(f"دقت مدل روی داده تست: {accuracy * 100:.2f}%")))
print(get_display(arabic_reshaper.reshape(f"خطای کلی: {loss:.4f}\n")))

# داده‌های جدید برای پیش‌بینی (سن و درآمد)
new_data = np.array([
    [28, 35],  # جوان با درآمد متوسط
    [50, 25]   # مسن با درآمد پایین
])

# پیش‌بینی احتمال خرید
predictions = model.predict(new_data)

# تبدیل احتمال به کلاس (با آستانه 0.5)
predicted_classes = (predictions > 0.5).astype(int)


for i in range(len(new_data)):
    print(get_display(arabic_reshaper.reshape(f"\nفرد با سن {new_data[i][0]} سال و درآمد {new_data[i][1]} میلیون:")))
    print(get_display(arabic_reshaper.reshape(f"  احتمال خرید: {predictions[i][0]:.2f}")))
    print(get_display(arabic_reshaper.reshape(f"  پیش‌بینی نهایی: {'خریدار' if predicted_classes[i][0] == 1 else 'غیرخریدار'}")))
