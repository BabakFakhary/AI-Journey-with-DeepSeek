#                                                                       به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install numpy scikit-learn matplotlib pandas tabulate
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# تولید داده‌های مصنوعی (مشکل غیرخطی)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# ---------------------------------------------------------------------------
# معماری شبکه: 2 ورودی -> 4 نورون (لایه پنهان) -> 1 خروجی
# X: آرایهٔ دو بعدی از مختصات نقاط (شکل: (n_samples, 2))
# y: برچسب‌های کلاس (0 یا 1) برای هر نقطه (شکل: (n_samples,))
# ---------------------------------------------------------------------------
class NeuralNetwork:
    def __init__(self):  
        self.W1 = np.random.randn(2, 4)   # دو سطر و چهار ستون عددی بین منهای یک و یک / وزن‌های لایه پنهان 
        self.b1 = np.zeros(4)             # یک آرایه یک‌بعدی با ۴ عنصر مقدار 0 ایجاد می‌کند  / بایاس لایه پنهان           
        self.W2 = np.random.randn(4, 1)   # چهار سطر و یک ستون عددی بین منهای یک و یک / وزن‌های لایه خروجی         
        self.b2 = np.zeros(1)             # بایاس لایه خروجی
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) # np.exp(10) => e^10
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1 # خروجی = تابع_فعال‌سازی(w1*x1 + w2*x2 + ... + wn*xn + bias) n=4 b=4
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def train(self, X, y, epochs=1000, lr=0.1):
        for epoch in range(epochs):
            # Forward Pass
            output = self.forward(X)
            
            # Backpropagation
            error = output - y.reshape(-1, 1)
            dW2 = np.dot(self.a1.T, error)
            db2 = np.sum(error, axis=0)
            dW1 = np.dot(X.T, (np.dot(error, self.W2.T) * (self.a1 * (1 - self.a1))))
            db1 = np.sum(np.dot(error, self.W2.T) * (self.a1 * (1 - self.a1)), axis=0)
            
            # به‌روزرسانی وزن‌ها
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            
            # نمایش خطا هر 100 دوره
            if epoch % 100 == 0:
                loss = np.mean(error ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---------------------------------------------------------------------------
# آموزش مدل
# ---------------------------------------------------------------------------
nn = NeuralNetwork()
nn.train(X_train, y_train)

# ---------------------------------------------------------------------------
# ارزیابی مدل
# ---------------------------------------------------------------------------
# خروجی nn.forward(X_test) عددی بین صفر و یک است
# astype(int):  مقادیر بولینی را تبدیل به صفر و یک می کند
y_pred = (nn.forward(X_test) > 0.5).astype(int)
# ابتدا آرایه چند بعدی را به یک بعدی تغییر داده و سپس آرایه بولینی می سازه که درست برابر یک و غلط به برابر صفر است سپس میانگین این آرایه محاسبه می گردد 
accuracy = np.mean(y_pred.flatten() == y_test) 
print(get_display(arabic_reshaper.reshape(f"دقت مدل روی داده تست: {accuracy * 100:.2f}%")))