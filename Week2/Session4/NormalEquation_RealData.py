#                                                                              به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# 1- پیاده‌سازی مدل با معادله نرمال
# -----------------------------------------------------------------------------

# 1. تولید داده نمونه
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 نمونه، 1 ویژگی
y = 4 + 3 * X + np.random.randn(100, 1)  # رابطه خطی با نویز

# 2. تقسیم داده به train/test (80% آموزش، 20% تست)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. اضافه کردن ستون 1 برای ضریب اریب (θ₀)
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]

# 4. محاسبه پارامترها با معادله نرمال
theta = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)

print(get_display(arabic_reshaper.reshape("پارامترهای یادگرفته شده:")))
print(get_display(arabic_reshaper.reshape(f"θ₀ (عرض از مبدأ): {theta[0][0]:.4f}")))
print(get_display(arabic_reshaper.reshape(f"θ₁ (شیب): {theta[1][0]:.4f}")))

# -----------------------------------------------------------------------------
# 2. ارزیابی مدل روی داده تست
# -----------------------------------------------------------------------------

# 1. آماده‌سازی داده تست
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

# 2. پیش‌بینی روی داده تست
y_pred = X_test_b.dot(theta)

# -----------------------------------------------------------------------------
# 3. محاسبه معیارهای ارزیابی
# -----------------------------------------------------------------------------

# یکی از معیارهای اصلی ارزیابی مدل‌های رگرسیون در یادگیری ماشین است
# هرچه کمتر بهتر: مقدار صفر نشان‌دهنده پیش‌بینی کاملًا دقیق است
mse = mean_squared_error(y_test, y_pred)

# یکی از مهم‌ترین معیارهای ارزیابی مدل‌های رگرسیون است
# ۱: مدل کاملًا دقیق (خطای صفر)
# ۰: مدل به خوبی مدل میانگین ساده عمل می‌کند
# منفی: مدل از یک مدل میانگین ساده هم بدتر است
r2 = r2_score(y_test, y_pred)

print(get_display(arabic_reshaper.reshape("\nارزیابی روی داده تست:")))
print(get_display(arabic_reshaper.reshape(f"MSE: {mse:.4f}")))
print(get_display(arabic_reshaper.reshape(f"R² Score: {r2:.4f}")))

# -----------------------------------------------------------------------------
# 4. مقایسه مقادیر واقعی و پیش‌بینی‌ شده
# -----------------------------------------------------------------------------

print(get_display(arabic_reshaper.reshape("\nمقایسه پیش‌بینی‌ها:")))
for i in range(5):  # نمایش 5 نمونه اول
    print(get_display(arabic_reshaper.reshape(f"واقعی: {y_test[i][0]:.2f} | پیش‌بینی: {y_pred[i][0]:.2f}")))

# -----------------------------------------------------------------------------
# 5. استفاده از مدل برای پیش‌بینی داده جدید
# -----------------------------------------------------------------------------


# داده جدید برای تست مدل
X_new = np.array([[0.5], [1.5]])  # 2 نمونه جدید
X_new_b = np.c_[np.ones((len(X_new), 1)), X_new]

# پیش‌بینی با مدل آموزش‌دیده
new_pred = X_new_b.dot(theta)
print(get_display(arabic_reshaper.reshape("\nپیش‌بینی برای داده جدید:")))
for x, y in zip(X_new, new_pred):
    print(get_display(arabic_reshaper.reshape(f"X = {x[0]:.1f} → پیش‌بینی y = {y[0]:.2f}")))