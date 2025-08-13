#                                                                              به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# داده‌های آموزشی
X_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y_train = np.array([2, 4, 5, 4, 5])

# داده‌های تست نمونه (ثابت)
X_test = np.array([1.5, 2.5, 3.5, 4.5]).reshape(-1, 1)
y_test = np.array([3, 4.2, 4.8, 4.5])  # مقادیر فرضی برای تست

# نرمال‌سازی داده‌ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ایجاد و آموزش مدل
model = SGDRegressor(
    learning_rate='constant',
    eta0=0.1,
    max_iter=100,
    tol=1e-3,
    random_state=42
)

# آموزش مدل
cost_history = []
for i in range(100):
    model.partial_fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_train_scaled)
    cost = mean_squared_error(y_train, y_pred_train)
    cost_history.append(cost)

# نمایش ضرایب نهایی
theta0 = model.intercept_[0]
theta1 = model.coef_[0]
print(get_display(arabic_reshaper.reshape("ضرایب نهایی (θ0, θ1):")), [theta0, theta1])

# پیش‌بینی روی داده‌های تست
y_pred_test = model.predict(X_test_scaled)

# نمایش نتایج پیش‌بینی و خطاها
print("\n" + get_display(arabic_reshaper.reshape("نتایج پیش‌بینی:")))
print("-" * 40)
for i in range(len(X_test)):
    error = abs(y_test[i] - y_pred_test[i])
    print(get_display(arabic_reshaper.reshape(f"نمونه {i+1}:")))
    print(get_display(arabic_reshaper.reshape(f"  X: {X_test[i][0]}, y واقعی: {y_test[i]:.2f}")))
    print(get_display(arabic_reshaper.reshape(f"  y پیش‌بینی شده: {y_pred_test[i]:.2f}")))
    print(get_display(arabic_reshaper.reshape(f"  خطا: {error:.2f}")))
    print("-" * 40)

# محاسبه خطای کلی (MSE)
mse = mean_squared_error(y_test, y_pred_test)
print("\n" + get_display(arabic_reshaper.reshape(f"خطای میانگین مربعات (MSE) روی داده تست: {mse:.4f}")))

# رسم نمودار هزینه
plt.plot(cost_history)
plt.xlabel(get_display(arabic_reshaper.reshape('تعداد تکرار')))
plt.ylabel(get_display(arabic_reshaper.reshape('هزینه (MSE)')))
plt.title(get_display(arabic_reshaper.reshape('تغییرات تابع هزینه در Gradient Descent')))
plt.show()

# رسم نمودار پیش‌بینی‌ها vs مقادیر واقعی
plt.scatter(X_test, y_test, color='blue', label=get_display(arabic_reshaper.reshape('مقادیر واقعی')))
plt.scatter(X_test, y_pred_test, color='red', label=get_display(arabic_reshaper.reshape('پیش‌بینی‌ها')))
plt.xlabel(get_display(arabic_reshaper.reshape('مقادیر X')))
plt.ylabel(get_display(arabic_reshaper.reshape('مقادیر y')))
plt.legend()
plt.title(get_display(arabic_reshaper.reshape('مقایسه پیش‌بینی‌ها با مقادیر واقعی')))
plt.show()

