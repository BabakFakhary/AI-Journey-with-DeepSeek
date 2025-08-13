#                                                                          به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install numpy scikit-learn matplotlib pandas
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# این کد یک درخت تصمیم بازگشتی ساده برای مسئله رگرسیون پیاده‌سازی می‌کند که می‌تواند رابطه بین درآمد و قیمت مسکن در کالیفرنیا را مدل کند.
# --------------------------------------------------------------------------------------

# بارگذاری دیتاست California Housing
data = fetch_california_housing()
X = data.data[:, 0]  # استفاده از ویژگی MedInc (میانگین درآمد)
y = data.target #  قیمت متوسط خانه‌های مسکونی در صد هزار دلار،توضیحات در فایل ورد

# تابع محاسبه خطای MSE
# def mse(y):
#     return np.mean((y - np.mean(y))**2)

# تابع بازگشتی ساخت درخت (نسخه ساده شده)
def fit(X, y, depth=3, min_samples=50):
    # شرایط توقف بهبود یافته
    if (depth == 0) or (len(y) < min_samples) or (np.std(y) < 0.1):
        # np.median(y) محاسبه میانه ، که نسبت میانگین مقاومت بیشتری دارد
        # میانه داده ها را از نزولی به صعودی مرتب کرده سپس عنصر وسط انتخاب می گردد 
        return {'value': np.median(y)}  # استفاده از میانه برای کاهش اثر outlierها
    
    # پیدا کردن بهترین تقسیم با درصدیل‌ها
    # محاسبه چهارک‌ها (Quartiles)
    percentiles = np.percentile(X, [25, 50, 75])
    
    best_split = None
    best_mse = float('inf') # برزگترین عدد اعشاری بی نهایت
    
    for percentile in percentiles:
        left_idx = X <= percentile
        right_idx = X > percentile
        
        if sum(left_idx) < min_samples or sum(right_idx) < min_samples:
            continue
            
        current_mse = np.var(y[left_idx]) + np.var(y[right_idx])
        
        if current_mse < best_mse:
            best_mse = current_mse
            best_split = {
                'threshold': percentile,
                'left': fit(X[left_idx], y[left_idx], depth-1),
                'right': fit(X[right_idx], y[right_idx], depth-1)
            }
    
    return best_split if best_split else {'value': np.median(y)}

# آموزش مدل با کنترل عمق
tree = fit(X, y, depth=3)
print(get_display(arabic_reshaper.reshape("درخت تصمیم با موفقیت ساخته شد!")))

# تابع predict (بدون تغییر)
def predict(tree, x):
    if 'value' in tree:
        return tree['value']
    if x <= tree['threshold']:
        return predict(tree['left'], x)
    else:
        return predict(tree['right'], x)

# ساخت نمونه 300 تایی بین ماکس و مین
X_test = np.linspace(X.min(), X.max(), 300) 

# تبدیل خروجی به آرایه با حلقه 
y_pred = [predict(tree, x) for x in X_test]

# رسم نمودار با جزئیات بیشتر
plt.figure(figsize=(12, 6))
# مصور سازی داده های واقعی
plt.scatter(X, y, alpha=0.2, label=get_display(arabic_reshaper.reshape('داده واقعی')))
# مصور سازی داده های پیشبینی
plt.plot(X_test, y_pred, 'r-', lw=3, label=get_display(arabic_reshaper.reshape('پیش‌بینی درخت')))
plt.xlabel(get_display(arabic_reshaper.reshape('درآمد متوسط (واحد: ده هزار دلار)')))
plt.ylabel(get_display(arabic_reshaper.reshape('قیمت خانه (واحد: صد هزار دلار)')))
plt.title(get_display(arabic_reshaper.reshape('رگرسیون درخت تصمیم - داده‌های California Housing')))
plt.legend()
plt.grid(True)

# نمایش آستانه‌های تقسیم
def plot_splits(tree, x_min, x_max):
    if 'threshold' in tree:
        plt.axvline(x=tree['threshold'], color='g', linestyle=':', alpha=0.5)
        plot_splits(tree['left'], x_min, tree['threshold'])
        plot_splits(tree['right'], tree['threshold'], x_max)

plot_splits(tree, X.min(), X.max())
plt.show()

# محاسبه خطا
y_train_pred = [predict(tree, x) for x in X]
mse = mean_squared_error(y, y_train_pred)
print(f"MSE: {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
# اگر R²
# برای محاسبه این حالت حتما باید تعداد پیشبینی و واقعی یکی باشد
#  بین ۰.۶ تا ۰.۸ باشد، مدل قابل قبول است.
#  اگر زیر ۰.۵ باشد، نیاز به بازنگری اساسی دارد.
np.random.seed(42)
indices = np.random.choice(len(y), size=300, replace=False)
y_true_sample = y[indices]
y_pred_sample = [predict(tree, X[i]) for i in indices]
print(get_display(arabic_reshaper.reshape(f"R² (بر روی نمونه ۳۰۰ تایی): {r2_score(y_true_sample, y_pred_sample):.4f}")))

