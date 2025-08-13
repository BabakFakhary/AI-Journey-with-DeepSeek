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

# داده‌های مصنوعی
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# نرمال‌سازی داده‌ها با StandardScaler (همان Z-Score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ایجاد و آموزش مدل SGDRegressor
model = SGDRegressor(
    learning_rate='constant',  # نرخ یادگیری ثابت
    eta0=0.1,                 # نرخ یادگیری اولیه
    max_iter=100,             # تعداد تکرارها
    tol=1e-3,                # حد تحمل برای توقف
    random_state=42
)

# آموزش مدل و ذخیره تاریخچه هزینه
cost_history = []
for i in range(100):
    model.partial_fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    cost = np.mean((y_pred - y)**2)  # محاسبه MSE
    cost_history.append(cost)

# نمایش نتایج
theta0 = model.intercept_[0]
theta1 = model.coef_[0]
print(get_display(arabic_reshaper.reshape("ضرایب نهایی (θ0, θ1):")), [theta0, theta1])

# رسم نمودار هزینه
plt.plot(cost_history)
plt.xlabel(get_display(arabic_reshaper.reshape('تعداد تکرار')))
plt.ylabel(get_display(arabic_reshaper.reshape('هزینه (MSE)')))
plt.title(get_display(arabic_reshaper.reshape('تغییرات تابع هزینه در Gradient Descent')))
plt.show()

