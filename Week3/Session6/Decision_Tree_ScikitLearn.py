#                                                                          به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install numpy scikit-learn matplotlib pandas tabulate
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# این کد یک درخت تصمیم برای مسئله رگرسیون پیاده‌سازی می‌کند که می‌تواند رابطه بین درآمد و قیمت مسکن در کالیفرنیا را مدل کند.
# --------------------------------------------------------------------------------------

# 1. Load California Housing dataset
# بارگذاری دیتاست California Housing
data = fetch_california_housing()
X = data.data[:, 0]  # استفاده از ویژگی MedInc (میانگین درآمد)
y = data.target #  قیمت متوسط خانه‌های مسکونی در صد هزار دلار،توضیحات در فایل ورد

# 2. Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. تغییر شکل داده‌های آموزشی و تست
# X.reshape(-1, 1) تغییر شکل آرایه به یک آرایه ستونی (تک ستونی)
X_train = X_train.reshape(-1, 1)  # تبدیل به (n_samples, 1)
X_test = X_test.reshape(-1, 1)    # تبدیل به (n_samples, 1)

# 4. Create and train the model
# 4. ایجاد و آموزش مدل
model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(X_train, y_train) 

# 5. Make predictions on test set
# 5. پیش‌بینی روی داده تست
y_pred = model.predict(X_test)

# 6. Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# 7. Visualize the tree
# 7. رسم درخت تصمیم
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=['MedInc'], 
          rounded=True, proportion=True)
plt.title("Decision Tree Structure" + get_display(arabic_reshaper.reshape("ساختار درخت تصمیم")))
plt.show()

# 8. Plot actual vs predicted values
# 8. مقایسه پیش‌بینی‌ها با داده واقعی
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label=get_display(arabic_reshaper.reshape('Actual Prices-مقادیر واقعی')))
plt.scatter(X_test, y_pred, color='red', alpha=0.5, label=get_display(arabic_reshaper.reshape('Predicted Prices-پیش‌ بینی‌ها')))
plt.xlabel(get_display(arabic_reshaper.reshape('Median Income (MedInc)-درآمد متوسط')))
plt.ylabel(get_display(arabic_reshaper.reshape('House Price (in $100,000)-قیمت خانه')))
plt.title(get_display(arabic_reshaper.reshape('Actual vs Predicted House Prices-مقایسه پیش‌بینی‌ها با داده واقعی')))
plt.legend()
plt.grid(True)
plt.show()

# 9. Plot prediction line (for continuous visualization)
# 9. رسم منحنی پیش‌بینی
X_grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1) # ساخت نمونه 300 تایی بین ماکس و مین
y_grid = model.predict(X_grid)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.3, label=get_display(arabic_reshaper.reshape('Actual Prices-مقادیر تست')))
plt.plot(X_grid, y_grid, color='red', linewidth=2, label=get_display(arabic_reshaper.reshape('Decision Tree Prediction-پیش‌ بینی مدل')))
plt.xlabel(get_display(arabic_reshaper.reshape('Median Income (MedInc)-درآمد متوسط')))
plt.ylabel(get_display(arabic_reshaper.reshape('House Price (in $100,000)-قیمت خانه')))
plt.title(get_display(arabic_reshaper.reshape('Decision Tree Regression-پیش‌ بینی درخت تصمیم')))
plt.legend()
plt.grid(True)
plt.show()

# 10. ایجاد داده‌های تست جدید و نمایش نتایج
new_test_samples = np.array([2.5, 3.0, 4.5, 5.0]).reshape(-1, 1)  # حتماً reshape شود
# مقادیر واقعی متناظر (اگر داشته باشیم)
actual_values = np.array([1.8, 2.3, 3.9, 4.2])  # مقادیر فرضی
# پیش‌بینی برای داده‌های جدید
predicted_values = model.predict(new_test_samples)

# نمایش نتایج به صورت جدول
results = pd.DataFrame({
    get_display(arabic_reshaper.reshape('MedInc (ویژگی)')): new_test_samples.flatten(),
    get_display(arabic_reshaper.reshape('قیمت واقعی (واحد: صد هزار دلار)')): actual_values,
    get_display(arabic_reshaper.reshape('پیش‌بینی مدل')): predicted_values,
    get_display(arabic_reshaper.reshape('خطا')): np.abs(actual_values - predicted_values)
})

print(results.to_markdown(index=False, floatfmt=".2f"))
