#                                                                             به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# داده‌های مصنوعی (X: ویژگی، y: برچسب)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# افزودن ضریب اریب (bias)
X_b = np.c_[np.ones((len(X), 1)), X]  # تبدیل X به شکل [1, X]

# محاسبه پارامترهای مدل (θ) با معادله نرمال
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# پیش‌بینی مدل
predictions = X_b.dot(theta)

# نمایش نتایج
print(get_display(arabic_reshaper.reshape("ضرایب مدل (θ0, θ1):")), theta)
plt.scatter(X, y, color='blue', label=get_display(arabic_reshaper.reshape('داده‌های واقعی')))
plt.plot(X, predictions, color='red', label=get_display(arabic_reshaper.reshape('پیش‌بینی مدل')))
plt.legend()
plt.show()