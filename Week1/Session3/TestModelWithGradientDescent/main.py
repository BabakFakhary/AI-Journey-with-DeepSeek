#                                                                به نام خدا
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import numpy as np
from linear_regression_gd import LinearRegressionGD
import matplotlib.pyplot as plt

# تولید داده‌های آموزشی
np.random.seed(42) # ثابت کردن اعداد تصادفی
X_train = 2 * np.random.rand(100, 1)
y_train = 4 + 3 * X_train + np.random.randn(100, 1)

# آموزش مدل
model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# نمایش پارامترهای آموزش دیده
print(get_display(arabic_reshaper.reshape(f"پارامترهای نهایی:\nθ₀ = {model.theta[0][0]:.4f}\nθ₁ = {model.theta[1][0]:.4f}")))
model.plot_cost_history()

# داده‌های تست جدید (می‌توانید مقادیر خود را جایگزین کنید)
X_test = np.array([[0.5], [1.5], [2.0]])  # ۳ نمونه جدید

# پیش‌بینی مدل
predictions = model.predict(X_test)

print(get_display(arabic_reshaper.reshape("\nپیش‌بینی‌ها برای داده‌های تست:")))
for x, y_pred in zip(X_test, predictions):
    print(get_display(arabic_reshaper.reshape(f"X = {x[0]:.2f} → پیش‌بینی y = {y_pred[0]:.2f}")))

# رسم نتایج
plt.scatter(X_train, y_train, alpha=0.7, label= get_display(arabic_reshaper.reshape('داده‌های آموزشی')))
plt.plot(X_test, predictions, 'ro-', label=get_display(arabic_reshaper.reshape('پیش‌بینی‌های تست')))
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()