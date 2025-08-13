#                                                                              به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# بارگذاری داده‌ها
data = fetch_california_housing()
X = data.data[:, 0:1]  # فقط یک ویژگی (متوسط تعداد اتاق‌ها)
y = data.target

# نرمال‌سازی داده‌ها
# یکی از مهم‌ترین ابزارهای پیش‌پردازش داده در یادگیری ماشین است
# هر ستون میانگین ≈ ۰ و انحراف معیار ≈ ۱ دارد
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# اضافه کردن ستون 1 برای θ0
X_b = np.c_[np.ones(len(X_scaled)), X_scaled]

# محاسبه پارامترها با معادلهٔ نرمال
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# پیش‌بینی
predictions = X_b.dot(theta)

# theta.ravel()  آرایه‌های چندبعدی را تبدیل به آرایه یک‌بعدی (مسطح‌شده) می‌کند
print(get_display(arabic_reshaper.reshape("ضرایب نهایی (θ0, θ1):")), theta.ravel())

# نمایش نتایج
plt.scatter(X, y, alpha=0.5, label=get_display(arabic_reshaper.reshape('داده‌های واقعی')))
plt.plot(X, predictions, color='red', label=get_display(arabic_reshaper.reshape('مدل رگرسیون با مدل معادله نرمال')))
plt.xlabel(get_display(arabic_reshaper.reshape('تعداد اتاق‌ها')))
plt.ylabel(get_display(arabic_reshaper.reshape('قیمت خانه (x1000$)')))
plt.legend()
plt.show()

