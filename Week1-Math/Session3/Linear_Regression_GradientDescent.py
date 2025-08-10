#                                                                             به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# 1. تولید داده‌های مصنوعی
np.random.seed(42) # ثابت کردن اعداد تصادفی
X = 2 * np.random.rand(100, 1)  # تولید ۱۰۰ عدد تصادفی بین بازه صفر تا دو
y = 4 + 3 * X + np.random.randn(100, 1)  # رابطه خطی با نویز

# 2. اضافه کردن ستون 1 برای ضریب اریب (bias term)
X_b = np.c_[np.ones((len(X), 1)), X]  # X_b.shape = (100, 2)

# 3. تنظیم پارامترهای Gradient Descent
learning_rate = 0.1  # نرخ یادگیری
n_iterations = 1000  # تعداد تکرارها
m = len(X_b)  # تعداد نمونه‌ها

# 4. مقداردهی اولیه پارامترها (تصادفی)
theta = np.random.randn(2, 1)  # شکل (2,1) برای θ₀ و θ₁

# 5. ذخیره تاریخچه تابع هزینه برای رسم نمودار
cost_history = []

# 6. حلقه اصلی Gradient Descent
for iteration in range(n_iterations):
    # محاسبه پیش‌بینی‌ها
    # y=Xb⋅θ
    predictions = X_b.dot(theta)  # X_b * theta
    
    # محاسبه خطاها
    errors = predictions - y
    
    # محاسبه گرادیان (مشتق تابع هزینه)
    gradients = 2/m * X_b.T.dot(errors)  # X_b.T * errors
    
    # به‌روزرسانی پارامترها
    theta = theta - learning_rate * gradients
    
    # محاسبه و ذخیره تابع هزینه (MSE)
    cost = (errors**2).mean()
    cost_history.append(cost)

# 7. نمایش نتایج
print(get_display(arabic_reshaper.reshape(f"پارامترهای نهایی:\nθ₀ (عرض از مبدأ): {theta[0][0]:.4f}\nθ₁ (شیب): {theta[1][0]:.4f}")))

# 8. رسم نمودار همگرایی
plt.figure(figsize=(12, 4))

# نمودار تابع هزینه در طول تکرارها
plt.subplot(1, 2, 1)
plt.plot(range(n_iterations), cost_history, 'b-')
plt.xlabel(get_display(arabic_reshaper.reshape('تعداد تکرارها')))
plt.ylabel(get_display(arabic_reshaper.reshape('تابع هزینه (MSE)')))
plt.title(get_display(arabic_reshaper.reshape('همگرایی Gradient Descent')))

# نمودار داده‌ها و خط رگرسیون
plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.7)
plt.plot(X, X_b.dot(theta), 'r-', linewidth=2)
plt.xlabel(get_display(arabic_reshaper.reshape('ویژگی (X)')))
plt.ylabel(get_display(arabic_reshaper.reshape('هدف (y)')))
plt.title(get_display(arabic_reshaper.reshape('خط رگرسیون نهایی')))

plt.tight_layout()
plt.show()