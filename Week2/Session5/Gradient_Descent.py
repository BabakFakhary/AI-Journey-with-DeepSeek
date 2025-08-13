#                                                                              به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# داده‌های مصنوعی
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# نرمال‌سازی داده‌ها (برای همگرایی بهتر)
# روش Z-Score Normalization
# الگوریتم‌هایی مانند رگرسیون خطی و شبکه‌های عصبی با داده‌های نرمال‌شده بهتر کار می‌کنند.
# np.mean(X) میانگین عناصر آرایه
# np.std(X)  انحراف معیار
X = (X - np.mean(X)) / np.std(X)

# اضافه کردن ستون 1 برای ضریب آزاد
X_b = np.c_[np.ones(len(X)), X]  

# پارامترهای مدل (θ0, θ1)
# عدد تصادفی بین منفی یک و یک
theta = np.random.randn(2, 1)

# تنظیمات Gradient Descent
learning_rate = 0.1
n_iterations = 100
cost_history = []

for iteration in range(n_iterations):
    gradients = 2/len(X) * X_b.T.dot(X_b.dot(theta) - y.reshape(-1, 1))
    theta -= learning_rate * gradients
    # کد تابع هزینه 
    # Mean Squared Error (MSE)
    cost = np.mean((X_b.dot(theta) - y.reshape(-1, 1))**2)
    cost_history.append(cost)

# نمایش نتایج
# theta.ravel()  آرایه‌های چندبعدی را تبدیل به آرایه یک‌بعدی (مسطح‌شده) می‌کند
print(get_display(arabic_reshaper.reshape("ضرایب نهایی (θ0, θ1):")), theta.ravel())
plt.plot(cost_history)
plt.xlabel(get_display(arabic_reshaper.reshape('تعداد تکرار')))
plt.ylabel(get_display(arabic_reshaper.reshape('هزینه (Cost)')))
plt.show()

