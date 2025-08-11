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
from sklearn.linear_model import LinearRegression

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
# Scikit-Learn
model = LinearRegression()
model.fit(X_scaled, y)

print(get_display(arabic_reshaper.reshape("ضرایب Scikit-Learn:")), model.intercept_, model.coef_)

# پیش‌بینی
predictions = model.predict(X_scaled)

# نمایش نتایج
plt.scatter(X, y, alpha=0.5, label=get_display(arabic_reshaper.reshape('داده‌های واقعی')))
plt.plot(X, predictions, color='red', label=get_display(arabic_reshaper.reshape('مدل رگرسیون با روش Scikit-Learn')))
plt.xlabel(get_display(arabic_reshaper.reshape('تعداد اتاق‌ها')))
plt.ylabel(get_display(arabic_reshaper.reshape('قیمت خانه (x1000$)')))
plt.legend()
plt.show()

